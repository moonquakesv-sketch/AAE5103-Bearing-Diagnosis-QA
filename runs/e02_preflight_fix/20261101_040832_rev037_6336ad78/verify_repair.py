"""Pure read-only rev037 integrity verifier. Standard library only; no project imports.
No subprocess, network, credentials, model execution, directory creation or writes.
"""
from pathlib import Path
import os,stat,json,hashlib,datetime,sys
ROOT=Path(__file__).absolute().parent
PROJECT=ROOT.parents[2]
BINDING_SHA="8128c19006074e106cb4f2a6e34f0d128e4b4fc4a8fa3e549a793c1b2d741f7a"
CONFIG_SHA="d286a74e8688b04b9b98c790f987381a903e529e65b652a8f87f6bf3cce0b0a5"
FLAGS=("formal_experiment_released","independent_cross_audit_accepted","paid_execution_authorized")
def guard(p):
 p=Path(p).absolute()
 assert p.is_relative_to(PROJECT) and ".." not in p.parts,str(p)
 for a in [*reversed(p.parents),p]: assert not a.is_symlink(),str(a)
 return p
def sha(p):
 h=hashlib.sha256()
 with os.fdopen(os.open(guard(p),os.O_RDONLY|os.O_NOFOLLOW),"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def read(p):
 with os.fdopen(os.open(guard(p),os.O_RDONLY|os.O_NOFOLLOW),"r",encoding="utf-8") as f:return json.load(f)
def scan(root,exclude=None):
 files={};links={}
 for base,dirs,names in os.walk(root,followlinks=False):
  for n in list(dirs)+names:
   p=Path(base)/n
   if p==exclude:
    if n in dirs:dirs.remove(n)
    continue
   s=p.lstat();rel=str(p.relative_to(root))
   if stat.S_ISLNK(s.st_mode):
    links[rel]={"target":os.readlink(p),"mode":s.st_mode,"dev":s.st_dev,"ino":s.st_ino,"mtime_ns":s.st_mtime_ns}
    if n in dirs:dirs.remove(n)
   elif stat.S_ISREG(s.st_mode):
    files[rel]={"sha256":sha(p),"size":s.st_size,"mode":s.st_mode,"dev":s.st_dev,"ino":s.st_ino,"mtime_ns":s.st_mtime_ns}
   else:assert stat.S_ISDIR(s.st_mode),str(p)
 return files,links
def history():
 base=read(ROOT/"baseline/workspace_start.json");current,links=scan(PROJECT,ROOT)
 assert links==base["links"],"old link membership, target or identity changed"
 assert not(set(base["files"])-set(current)),"old file missing"
 changed=sorted(p for p,v in base["files"].items() if current[p]!=v)
 added=sorted(set(current)-set(base["files"]))
 c=read(ROOT/"compatibility.json");plan=c["only_preexisting_change"]
 assert changed==[plan] and added==[c["archive_path"]],(changed,added)
 assert current[plan]["sha256"]==c["plan_after_sha256"]
 assert sha(PROJECT/c["archive_path"])==base["files"][plan]["sha256"]==c["archive_sha256"]==sha(ROOT/"before"/plan)
 assert all(current[plan][k]==base["files"][plan][k] for k in ("mode","dev","ino"))
 return {"old_files":len(base["files"]),"old_links":len(base["links"]),"changed":changed,"outside_additions":added,"all_other_old_identity_and_bytes_preserved":True}
def readiness():
 b=read(ROOT/"config/execution_binding.json");cfg=read(ROOT/"config/execution.json")
 assert sha(ROOT/"config/execution_binding.json")==BINDING_SHA
 assert len(b["files"])==82
 for rel,h in b["files"].items():assert sha(PROJECT/rel)==h,rel
 cfgsha=hashlib.sha256(json.dumps(cfg,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
 assert cfgsha==CONFIG_SHA==b["configuration_sha256"]
 assert b["original77_not_rebuilt"] and b["previous82_not_rebuilt"]
 for name in FLAGS:assert b[name] is False
 assert cfg["formal_experiment_released"] is False and cfg["user_approved_hard_budget"] is False
 a=read(ROOT/"validation/repair_acceptance.json")
 assert a["eligible_for_user_formal_release_step"] and a["no_open_implementation_blocker_in_checked_scope"]
 assert a["E02_started"] is False and a["project_model_API_calls"]==0
 for name in FLAGS:assert a[name] is False
 for v in a["peer_originals"].values():assert sha(ROOT/v["path"])==v["sha256"]
 d=read(ROOT/"release/reviewed_authorization_FALSE.json")
 for name in FLAGS:assert d[name] is False
 assert d["preparation_binding_sha256"]==BINDING_SHA and d["configuration_sha256"]==CONFIG_SHA
 assert d["scope"]==cfg["scope"] and d["hard_budget_USD"]=="20.00" and d["max_calls"]==364
 assert d["user_authorization_record"] is None and d["user_authorization_sha256"] is None and d["authorization_id"] is None
 assert sha(PROJECT/d["cross_audit_report_path"])==d["cross_audit_report_sha256"]
 assert not guard(PROJECT/d["batch_output"]).exists(),"proposed E02 batch unexpectedly exists"
 for name,n in [("rev037_oracles_after.json",8),("current_regressions.json",31)]:
  t=read(ROOT/"validation"/name);assert t["success"] and t["tests"]==n and not t["failures"] and not t["errors"]
 cl=read(ROOT/"validation/repair_closure.json");assert cl["all_current_four_closed"] and len(cl["items"])==4 and cl["latent_C004_I_NA_also_fixed"]
 plan=guard(PROJECT/"AAE5103_项目计划与交接.md").read_text()
 assert "v1.35" in plan and "rev037" in plan
 expires=datetime.datetime.fromisoformat(cfg["price"]["checked_at"])+datetime.timedelta(hours=cfg["price"]["maximum_age_hours"])
 return {"binding_count":82,"binding_sha256":BINDING_SHA,"configuration_sha256":CONFIG_SHA,"technical_ready_for_user_release":True,"paid_authorization_present":False,"E02_started":False,"price_route_expires_JST":expires.astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),"price_route_currently_fresh":datetime.datetime.now(datetime.timezone.utc)<=expires}
def seal():
 f=read(ROOT/"final_verification.json");m=read(ROOT/"self_files.json")
 files,links=scan(ROOT)
 assert set(files)==set(m["complete_expected_files"]),"own exact file membership differs"
 assert links==read(ROOT/"links.json")["links"]==m["links"],"own exact link identities differ"
 for rel,v in m["files"].items():assert files[rel]==v,rel
 lines=guard(ROOT/"deliverables.sha256").read_text().splitlines();listed={}
 for line in lines:
  h,rel=line.split("  ",1);assert rel not in listed
  listed[rel]=h
 assert set(listed)==set(files)-{"deliverables.sha256","final_verification.json"}
 for rel,h in listed.items():assert files[rel]["sha256"]==h,rel
 assert sha(ROOT/"deliverables.sha256")==f["deliverables_sha256"]
 assert sha(ROOT/"self_files.json")==f["self_files_sha256"]
 assert len(files)==f["own_regular_files"] and len(links)==f["own_links"]
 assert f["technical_ready_for_user_release"] and f["E02_started"] is False
 for name in FLAGS:assert f[name] is False
 return {"own_regular_files":len(files),"own_links":len(links),"deliverables_sha256":sha(ROOT/"deliverables.sha256"),"final_verification_sha256":sha(ROOT/"final_verification.json"),"all_own_hashes_and_recorded_identities_match":True}
if __name__=="__main__":
 assert sys.argv[1:] in ([],["--preseal"])
 result={"status":"PASS","checked_at_JST":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),"history":history(),"readiness":readiness(),"pure_read_only_standard_library_no_project_imports":True}
 if not sys.argv[1:]:result["evidence_seal"]=seal()
 print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
