use super::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sp1_sdk::{SP1Proof, SP1ProofWithPublicValues, SP1PublicValues};

fn approval(p: Program, episodes: u32) -> Approval {
    Approval { schema:"zkdrl-ctc-approval-v1".into(), run_id:"run-1".into(), group_id:"group-1".into(),
        algorithm:p.id().into(), elf_sha256:sha(p.elf()), vk_bytes32:format!("0x{}", "00".repeat(32)),
        config_commit:[11;32], challenge:[23;32], kind:1, episodes,
        scale:if p == Program::Dqn {16} else {65536}, width:p.width(), capacity:p.capacity() }
}
fn journal(p: Program, a: &Approval) -> Value {
    let slots = a.episodes*6;
    if p == Program::Dqn {
        let updates = slots.saturating_sub(31);
        json!({"schema":202,"config_commit":a.config_commit,"final_commit":vec![0u8;32],"challenge":a.challenge,
            "kind":a.kind,"scale":a.scale,"slots":slots,"train":1,"return_units":0,
            "interactions":slots,"updates":updates,"resets":a.episodes,"padded":0,
            "target_copies":updates/32,"replay_wraps":slots.saturating_sub(zkdrl_ctc_dqn::CAP as u32)})
    } else {
        json!({"schema":203,"config_commit":a.config_commit,"initial_commit":vec![0u8;32],"final_commit":vec![0u8;32],"challenge":a.challenge,
            "kind":a.kind,"episodes":a.episodes,"slots":slots,"interactions":slots,"updates":a.episodes,
            "resets":a.episodes,"padded":0,"successes":0,"return_units":0})
    }
}
fn typed_journal(p: Program, v: &Value) -> Result<Journal> { Journal::from_json(p, &serde_json::to_vec(v)?) }
fn dqn_input() -> zkdrl_ctc_dqn::PrivateInput {
    zkdrl_ctc_dqn::PrivateInput { schema:202, kind:1, scale:16, slots:12, train:1, seed:30,
        rng_seed:[7;32], initial_coarse:vec![0;zkdrl_ctc_dqn::P], params:[16,0,-16,8],
        config_salt:[11;32], final_salt:[19;32], challenge:[23;32] }
}
fn a2c_input() -> zkdrl_ctc_a2c::PrivateInput {
    zkdrl_ctc_a2c::PrivateInput { schema:203, kind:1, episodes:2, seed:30,
        rng_seed:[7;32], initial_actor:vec![0;zkdrl_ctc_a2c::ACTOR], initial_critic:vec![0;zkdrl_ctc_a2c::CRITIC],
        config_salt:[11;32], final_salt:[19;32], challenge:[23;32] }
}

#[test]
fn exact_cli_and_cuda_modes() {
    for args in ["host identity dqn", "host native a2c in", "host native dqn in --capture",
        "host execute a2c in", "host prove dqn in approval proof", "host verify dqn proof approval",
        "host tamper a2c proof approval", "host audit a2c proof approval"] {
        assert!(Command::parse(&args.split_whitespace().map(str::to_owned).collect::<Vec<_>>()).is_ok(), "{args}");
    }
    for args in ["host", "host identity dqn extra", "host native dqn in --n2", "host native dqn in --capture extra",
        "host execute dqn in expected", "host prove dqn in approval", "host verify dqn proof approval expected",
        "host native unknown in", "host native-audit dqn in", "host mock dqn"] {
        assert!(Command::parse(&args.split_whitespace().map(str::to_owned).collect::<Vec<_>>()).is_err(), "{args}");
    }
    cuda_environment_values("cuda", "0", "GPU-authorized").unwrap();
    for mode in ["cpu", "mock", "network", "", "CUDA", "cuda "] { assert!(cuda_environment_values(mode,"0","GPU-x").is_err()); }
    for visible in ["", "-1", "1", "0,1", " 0", "GPU-x"] { assert!(cuda_environment_values("cuda",visible,"GPU-x").is_err()); }
    assert!(cuda_environment_values("cuda","0"," \n").is_err());
    for name in ["h32", "h64", "c128", "c256", "dqn-h32", "dqn-h64", "dqn-c128", "dqn-c256"] {
        assert!(Program::parse(name).is_err(), "profiles are selected by the compiled binary: {name}");
    }
}

#[test]
fn compiled_profile_is_derived_from_common() {
    let expected = if cfg!(feature = "h32") { (32, Some(64)) }
        else if cfg!(feature = "h64") { (64, Some(64)) }
        else if cfg!(feature = "c128") { (16, Some(128)) }
        else if cfg!(feature = "c256") { (16, Some(256)) }
        else { (16, Some(64)) };
    assert_eq!((zkdrl_ctc_dqn::H as u32, Some(zkdrl_ctc_dqn::CAP as u32)), expected);
    assert_eq!((Program::Dqn.width(), Program::Dqn.capacity()), expected);
    assert_eq!(Program::Dqn.profile(), json!({"width":expected.0,"capacity":expected.1}));
    assert_eq!(Program::A2c.width(), zkdrl_ctc_a2c::H as u32);
    assert_eq!(Program::A2c.profile(), json!({"width":16,"capacity":null}));
}

#[test]
fn approvals_reject_other_and_crossed_profiles() {
    for p in [Program::Dqn, Program::A2c] {
        let a = approval(p,100);
        let j = typed_journal(p,&journal(p,&a)).unwrap();
        // Hold ELF/key and all other approval fields fixed to isolate profile policy.
        for width in [16,32,64] {
            for capacity in [None,Some(64),Some(128),Some(256)] {
                let mut candidate = a.clone();
                candidate.width = width;
                candidate.capacity = capacity;
                let matches = (width,capacity) == (p.width(),p.capacity());
                assert_eq!(Approval::parse(p,&serde_json::to_vec(&candidate).unwrap()).is_ok(), matches);
                assert_eq!(j.policy(p,&candidate).is_ok(), matches);
            }
        }
    }
}

#[test]
fn dqn_approval_counters_follow_common_ring_capacity() {
    let p = Program::Dqn;
    let cap = zkdrl_ctc_dqn::CAP as u32;
    assert_eq!(zkdrl_ctc_dqn::WARMUP,32);
    assert_eq!(zkdrl_ctc_dqn::TARGET_EVERY,32);
    let mut a = approval(p,(cap+6)/6);
    a.kind = 0; // Login padding lets active interactions straddle the ring boundary.
    for (interactions, wraps) in [(cap-1,0),(cap,0),(cap+1,1)] {
        let mut value = journal(p,&a);
        value["interactions"] = json!(interactions);
        value["padded"] = json!(6*a.episodes-interactions);
        value["updates"] = json!(interactions.saturating_sub(31));
        value["target_copies"] = json!(interactions.saturating_sub(31)/32);
        value["replay_wraps"] = json!(wraps);
        typed_journal(p,&value).unwrap().policy(p,&a).unwrap();
        value["replay_wraps"] = json!(wraps+1);
        assert!(typed_journal(p,&value).unwrap().policy(p,&a).is_err());
    }
    // All candidate capacities have wrapped: another profile's count must fail.
    let a = approval(p,100);
    for other_cap in [64,128,256] {
        let mut value = journal(p,&a);
        value["replay_wraps"] = json!(600-other_cap);
        assert_eq!(typed_journal(p,&value).unwrap().policy(p,&a).is_ok(), other_cap == cap);
    }
}

#[test]
fn canonical_journal_abis_remain_148_and_172_bytes() {
    for (p,size,schema) in [(Program::Dqn,148,202u32),(Program::A2c,172,203u32)] {
        let a = approval(p,100);
        let bytes = typed_journal(p,&journal(p,&a)).unwrap().encode().unwrap();
        assert_eq!(p.journal_size(),size);
        assert_eq!(bytes.len(),size);
        assert_eq!(&bytes[..4],&schema.to_le_bytes());
        let decoded = Journal::decode(p,&bytes).unwrap();
        assert_eq!(decoded.encode().unwrap(),bytes);
        decoded.policy(p,&a).unwrap();
        let other = if p == Program::Dqn { Program::A2c } else { Program::Dqn };
        assert!(Journal::decode(other,&bytes).is_err());
    }
}

#[test]
fn approval_schema_metadata_and_canonical_encodings() {
    for p in [Program::Dqn, Program::A2c] {
        let a = approval(p,4000);
        a.validate(p).unwrap();
        a.pin_key(&a.vk_bytes32).unwrap();
        assert!(a.pin_key(&format!("0x{}", "01".repeat(32))).is_err());
        let v = serde_json::to_value(&a).unwrap();
        let s = serde_json::to_string(&a).unwrap();
        Approval::parse(p,s.as_bytes()).unwrap();
        assert!(Approval::parse(p,format!("{{\"kind\":1,{}", &s[1..]).as_bytes()).is_err());
        for key in v.as_object().unwrap().keys() {
            let mut bad = v.clone(); bad.as_object_mut().unwrap().remove(key);
            assert!(Approval::parse(p,&serde_json::to_vec(&bad).unwrap()).is_err(), "missing {key}");
        }
        for (key, value) in [
            ("extra",json!(0)),("schema",json!("wrong")),("run_id",json!("")),("group_id",json!("x".repeat(257))),
            ("algorithm",json!("other")),("kind",json!(2)),("episodes",json!(0)),("episodes",json!(4001)),
            ("episodes",json!(-1)),("width",json!(0)),("capacity",json!(999)),("scale",json!(17)),
            ("elf_sha256",json!("AB".repeat(32))),("elf_sha256",json!("00".repeat(32))),
            ("vk_bytes32",json!(format!("0x{}","AA".repeat(32)))),("vk_bytes32",json!("0x00")),
            ("config_commit",json!(vec![0;31])),("challenge",json!(vec![256;32])),
        ] {
            let mut bad = v.clone(); bad[key] = value;
            assert!(Approval::parse(p,&serde_json::to_vec(&bad).unwrap()).is_err(), "{key}");
        }
    }
    assert_eq!(Program::Dqn.width(), zkdrl_ctc_dqn::H as u32);
    assert_eq!(Program::Dqn.capacity(), Some(zkdrl_ctc_dqn::CAP as u32));
    assert_eq!((Program::A2c.width(),Program::A2c.capacity()),(16,None));
}

#[test]
fn journal_policy_and_output_freedom() {
    for p in [Program::Dqn,Program::A2c] {
        for n in [1,2,1000,4000] {
            let a = approval(p,n);
            let value = journal(p,&a);
            let j = typed_journal(p,&value).unwrap();
            j.policy(p,&a).unwrap();
            let bytes = j.encode().unwrap();
            assert_eq!(bytes.len(),p.journal_size());
            Journal::decode(p,&bytes).unwrap().policy(p,&a).unwrap();
            assert!(Journal::decode(p,&bytes[..bytes.len()-1]).is_err());
            let mut extra = bytes.clone(); extra.push(0);
            assert!(Journal::decode(p,&extra).is_err());
            for (key, old) in value.as_object().unwrap() {
                let mut changed = value.clone();
                if old.is_array() { changed[key][0] = json!(old[0].as_u64().unwrap() ^ 1); }
                else if key == "return_units" { changed[key] = json!(i64::MIN); }
                else { changed[key] = json!(old.as_u64().unwrap() ^ 1); }
                let result = typed_journal(p,&changed).unwrap().policy(p,&a);
                if ["return_units","final_commit","initial_commit"].contains(&key.as_str()) {
                    assert!(result.is_ok(), "computed output must not be preknown: {key}");
                } else { assert!(result.is_err(), "unchecked policy field: {key}"); }
            }
            let mut overflow = value.clone(); overflow["interactions"] = json!(u32::MAX); overflow["padded"] = json!(1);
            assert!(typed_journal(p,&overflow).unwrap().policy(p,&a).is_err());
            let s = serde_json::to_string(&value).unwrap();
            assert!(Journal::from_json(p,format!("{{\"kind\":1,{}",&s[1..]).as_bytes()).is_err());
            let mut extra = value.clone(); extra["extra"] = json!(0);
            assert!(typed_journal(p,&extra).is_err());
            for key in value.as_object().unwrap().keys() {
                let mut missing = value.clone(); missing.as_object_mut().unwrap().remove(key);
                assert!(typed_journal(p,&missing).is_err());
            }
        }
        // Login padding is allowed, with counters based on active interactions.
        let mut a = approval(p,100); a.kind = 0;
        let mut value = journal(p,&a);
        value["interactions"] = json!(100); value["padded"] = json!(500);
        if p == Program::Dqn {
            value["updates"] = json!(69); value["target_copies"] = json!(2);
            value["replay_wraps"] = json!(100u32.saturating_sub(zkdrl_ctc_dqn::CAP as u32));
        } else { value["successes"] = json!(100); }
        typed_journal(p,&value).unwrap().policy(p,&a).unwrap();
        if p == Program::A2c {
            value["successes"] = json!(101);
            assert!(typed_journal(p,&value).unwrap().policy(p,&a).is_err());
        }
    }
}

#[test]
fn typed_input_limits_and_approval_binding_without_training() {
    let di = dqn_input();
    let ai = a2c_input();
    for (p,v) in [(Program::Dqn,serde_json::to_value(&di).unwrap()),(Program::A2c,serde_json::to_value(&ai).unwrap())] {
        Input::parse(p,&serde_json::to_vec(&v).unwrap()).unwrap();
        let s = serde_json::to_string(&v).unwrap();
        assert!(Input::parse(p,format!("{{\"kind\":1,{}",&s[1..]).as_bytes()).is_err());
        let mut bad = v.clone(); bad["extra"] = json!(0);
        assert!(Input::parse(p,&serde_json::to_vec(&bad).unwrap()).is_err());
        for key in v.as_object().unwrap().keys() {
            let mut bad = v.clone(); bad.as_object_mut().unwrap().remove(key);
            assert!(Input::parse(p,&serde_json::to_vec(&bad).unwrap()).is_err());
        }
        for key in ["schema","kind",if p == Program::Dqn {"slots"} else {"episodes"}] {
            let mut bad = v.clone(); bad[key] = json!(u32::MAX);
            assert!(Input::parse(p,&serde_json::to_vec(&bad).unwrap()).is_err());
        }
    }
    for bound in [i64::MIN,-(1<<20)-1,1<<20,i64::MAX] {
        let mut bad = di.clone(); bad.initial_coarse[0] = bound;
        assert!(Input::Dqn(bad).validate().is_err());
    }
    for slots in [0,5,7,24001,u32::MAX] {
        let mut bad = di.clone(); bad.slots = slots;
        assert!(Input::Dqn(bad).validate().is_err());
    }
    let mut bad = di.clone(); bad.kind = 0;
    assert!(Input::Dqn(bad).validate().is_err()); // Login forbids nonzero reward parameters.
    for bound in [i64::MIN,-(256i64<<40),-(256i64<<40)+1,256i64<<40,i64::MAX] {
        let mut bad = ai.clone(); bad.initial_actor[0] = bound;
        assert!(Input::A2c(bad).validate().is_err());
    }
    for (p,i,commit) in [(Program::Dqn,Input::Dqn(di.clone()),hash(&zkdrl_ctc_dqn::config_bytes(&di))),
        (Program::A2c,Input::A2c(ai.clone()),hash(&zkdrl_ctc_a2c::config_bytes(&ai)))] {
        let mut a = approval(p,2); a.config_commit = commit;
        i.check_approval(&a).unwrap();
        a.challenge[0] ^= 1;
        assert!(i.check_approval(&a).is_err());
        a.challenge[0] ^= 1; a.config_commit[0] ^= 1;
        assert!(i.check_approval(&a).is_err());
        a.config_commit = commit; a.episodes += 1;
        assert!(i.check_approval(&a).is_err());
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct DecodeFixture { flag: bool, count: u32, bytes: Vec<u8> }
#[test]
fn strict_bincode_limits_aliases_and_non_groth16() {
    let fixture = DecodeFixture { flag:true,count:24000,bytes:vec![1,2,3] };
    let bytes = bincode::serialize(&fixture).unwrap();
    assert_eq!(proof::strict_decode::<DecodeFixture>(&bytes).unwrap(),fixture);
    for n in 0..bytes.len() { assert!(proof::strict_decode::<DecodeFixture>(&bytes[..n]).is_err()); }
    let mut extra = bytes.clone(); extra.push(0);
    assert!(proof::strict_decode::<DecodeFixture>(&extra).is_err());
    let mut alias = bytes.clone(); alias[0] = 2;
    assert!(proof::strict_decode::<DecodeFixture>(&alias).is_err());
    assert!(proof::strict_decode::<Vec<u8>>(&u64::MAX.to_le_bytes()).is_err());
    assert!(proof::strict_decode::<Vec<u8>>(&vec![0;proof::MAX_BUNDLE as usize+1]).is_err());
    let value = json!({"proof":SP1Proof::Core(vec![]),"public_values":SP1PublicValues::from(&vec![0u8;148]),
        "sp1_version":CIRCUIT,"tee_proof":null});
    let p: SP1ProofWithPublicValues = serde_json::from_value(value).unwrap();
    assert!(proof::decode(&bincode::serialize(&p).unwrap()).is_err());
}
