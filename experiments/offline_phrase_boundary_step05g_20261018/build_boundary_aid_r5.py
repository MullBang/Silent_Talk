"""Build a single-item visual review aid from the unchanged r4 media allowlist.

Markers represent the confirmed audio interval, not true articulation bounds.
Browser frame navigation targets the center of an existing25fps display bin.
No new media, saved judgement, mutable UI pointer or historical file is written.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OLD=ROOT/"experiments/offline_phrase_review_step05f_20261018"
PREVIOUS_UI=BASE/"ui/r4/manifest.json"
QUESTION=BASE/"question_batches/008_all8_visual.json"
EVIDENCE=BASE/"user_evidence/014_all8_visual_inconclusive.json"
SUMMARY=BASE/"results/stage5g_summary_r5.json"
OUTPUT=BASE/"ui/r5"


def require(condition: bool,message: str) -> None:
    """Fail explicitly if the bound review scope or source clocks differ."""
    if not condition: raise ValueError(message)


def bind(path: Path) -> dict[str,Any]:
    """Read file identity without modifying its bytes."""
    with path.open("rb") as stream: digest=hashlib.file_digest(stream,"sha256").hexdigest()
    return {"path":str(path.resolve()),"sha256":digest,"bytes":path.stat().st_size}


def read(path: Path) -> Any:
    """Read JSON evidence as data."""
    return json.loads(path.read_text(encoding="utf-8"))


def checked(spec: dict[str,Any]) -> Path:
    """Check the exact local binding used by this screen."""
    path=Path(spec["path"])
    require(bind(path)=={k:spec[k] for k in ("path","sha256","bytes")},"Review input changed: "+str(path))
    return path


def fraction(value: dict[str,Any]) -> Fraction:
    """Preserve rational source clocks without floating-point boundary rounding."""
    require(type(value["numerator"]) is int and type(value["denominator"]) is int and value["denominator"]>0,"Invalid exact clock")
    return Fraction(value["numerator"],value["denominator"])


def rational(value: Fraction) -> dict[str,int|float]:
    """Keep exact source evidence alongside browser display seconds."""
    return {"numerator":value.numerator,"denominator":value.denominator,"seconds":float(value)}


def write(path: Path,value: Any) -> None:
    """Create only new artifacts in the assigned r5 directory."""
    require(path.parent==OUTPUT and not path.exists(),"Unsafe/existing output")
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


PAGE=r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>입술 경계 살펴보기 · 한 구절씩</title>
<style>
:root{font-family:"Malgun Gothic",system-ui,sans-serif;color:#193445;background:#edf2f6;font-synthesis:none}*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:22px 24px 52px}h1{font-size:25px;margin:0 0 10px}h2{font-size:23px;margin:0}h3{font-size:16px;margin:0 0 8px}p{line-height:1.65;margin:8px 0}.intro{font-size:14px;color:#4b6474}.status{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.badge{font-size:12px;background:#fff;border:1px solid #c7d6e0;border-radius:20px;padding:6px 11px}.notice{border-left:4px solid #c19442;background:#fff9ed;padding:11px 14px;font-size:13px;border-radius:6px}.selectbar{display:flex;justify-content:space-between;align-items:end;gap:15px;margin:20px 0 12px}label{font-size:13px}select{display:block;margin-top:6px;padding:8px 10px;border:1px solid #aebfca;border-radius:7px;background:white;color:#16364b;font:inherit;max-width:100%}.phraseHeading{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:15px 0}.qid{font-size:14px;background:#215e79;color:#fff;border-radius:6px;padding:6px 9px}.hint{font-size:13px;color:#4a6575}.compare{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:16px;align-items:start}.panel{background:#fff;padding:18px;border:1px solid #cbd9e2;border-radius:12px;min-width:0}.primary{border:2px solid #6c9ab0;padding:17px}.videoBox{background:#142a38;border-radius:8px;padding:8px;margin:12px 0}.videoBox video{display:block;width:100%;max-width:410px;aspect-ratio:1;margin:auto;border-radius:4px;object-fit:contain}.reference video{max-width:340px}.position{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:13px;margin:10px 0}.region{font-weight:bold;border-radius:6px;padding:6px 10px;background:#edf1f5;color:#526e7f}.region.inside{background:#e1f1ea;color:#16634d}.time{font-variant-numeric:tabular-nums;color:#4b6373}.controls{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:10px 0}button{border:1px solid #b7cad6;background:#f5f8fa;color:#24495e;border-radius:7px;padding:9px 12px;font:inherit;font-size:13px;cursor:pointer}button:hover{background:#e6eff5}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #e4ab42;outline-offset:2px}button.strong{background:#225e79;color:white;border-color:#225e79}button:disabled{opacity:.5;cursor:wait}.speed{display:flex;gap:5px;align-items:center}.speed select{display:inline-block;margin:0;padding:7px;font-size:13px}.timeline{position:relative;margin:30px 2px 6px;padding-top:16px;height:50px}.track{position:absolute;left:0;right:0;top:16px;height:9px;background:#d8e1e8;border-radius:6px}.audioRange{position:absolute;top:0;height:9px;background:#65aa91}.marker{position:absolute;top:-10px;width:2px;height:29px;background:#186c58;z-index:1}.marker span{position:absolute;white-space:nowrap;top:-18px;left:0;transform:translateX(-50%);font-size:11px;font-weight:bold;color:#17634d}.timeline input{position:relative;display:block;width:100%;height:28px;margin:0;accent-color:#184f6e;z-index:2;cursor:pointer}.legend{font-size:12px;color:#42695a;margin:3px 0}.frameInfo{font-size:12px;color:#567080;min-height:20px}.small{font-size:12px;color:#597180}.quick{display:flex;gap:7px;flex-wrap:wrap}.reference p{font-size:13px;color:#536e7c}.audioSection{margin-top:15px;padding:12px 16px;background:#fff;border:1px solid #ccdbe4;border-radius:10px}summary{cursor:pointer;font-size:14px;color:#2d5d74}.audioGrid{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:12px}audio{display:block;width:100%;margin:7px 0}.footerNote{margin-top:16px;font-size:13px;color:#4f6b7b}.error{color:#a22d32;font-size:13px;min-height:20px;margin-top:8px}@media(max-width:680px){main{padding:18px 12px 40px}.compare{grid-template-columns:1fr}.selectbar{align-items:start;flex-direction:column}.panel{padding:13px}.audioGrid{grid-template-columns:1fr}.videoBox video{max-width:340px}.reference video{max-width:280px}h1{font-size:22px}h2{font-size:20px}}
</style></head><body><main>
<h1>입술 경계 살펴보기</h1><p class="intro">한 구절씩 천천히 봅니다. 먼저 Q01의 <b>앞뒤 문맥 영상</b>에서 시작과 끝 주변을 살펴보세요.</p>
<div class="status"><span class="badge">음성 8구절 확인됨</span><span class="badge">입술 경계: 8구절 모두 구분 어려움 · 확인 처리 0</span></div>
<p class="notice">표시된 선은 <b>확인한 음성 범위</b>이며, 입 움직임의 정답 경계를 뜻하지 않습니다. 입이 완전히 닫혀야 하는 것은 아닙니다. 확실하지 않으면 판단 어려움으로 남겨도 됩니다.</p>
<div class="selectbar"><label for="item">살펴볼 구절<select id="item" aria-label="살펴볼 구절"></select></label><span class="hint">이 화면에는 현재 선택한 한 구절만 표시됩니다.</span></div>
<div class="phraseHeading"><span class="qid" id="qid">Q01</span><h2 id="phraseTitle"></h2></div>
<div class="compare">
<section class="panel primary" aria-labelledby="contextTitle"><h3 id="contextTitle">앞뒤 문맥 영상 · 여기서 천천히 탐색</h3><p class="hint">초록 범위 전후로 입 움직임이 어떻게 이어지는지 봅니다.</p>
<div class="videoBox"><video id="contextVideo" muted playsinline preload="metadata" aria-label="앞뒤 문맥 무음 영상"></video></div>
<div class="position"><span id="region" class="region">영상 준비 중</span><span class="time" id="positionTime"></span></div>
<div class="controls"><button id="playContext" class="strong" type="button">문맥 재생</button><label class="speed" for="speed">재생 속도<select id="speed"><option value="0.5" selected>0.5배</option><option value="1">1배</option><option value="0.25">0.25배</option></select></label></div>
<div class="quick"><button id="startPart" type="button">시작 부분 보기</button><button id="endPart" type="button">끝 부분 보기</button></div>
<div class="timeline" aria-label="확인한 음성 범위가 표시된 문맥 타임라인"><div class="track"><div id="audioRange" class="audioRange"></div><div id="startMarker" class="marker"><span>음성 시작</span></div><div id="endMarker" class="marker"><span>음성 끝</span></div></div><input id="frameSlider" aria-label="문맥 영상 표시 프레임 탐색" type="range" min="0" max="1" value="0" step="1"></div>
<p class="legend">초록 표시 = 확인한 음성 범위</p>
<div class="controls"><button id="prevFrame" type="button">← 1프레임</button><button id="nextFrame" type="button">1프레임 →</button><button id="jumpStart" type="button">음성 시작 근처</button><button id="jumpEnd" type="button">음성 끝 근처</button></div>
<p id="frameInfo" class="frameInfo"></p><p id="markerTimes" class="small"></p>
<p class="small">프레임 이동은 이 문맥 영상의 25fps 표시 칸 중심으로 탐색합니다. 브라우저 재생만으로 원본의 정밀한 시각 경계를 보장하지는 않습니다.</p>
</section>
<section class="panel reference" aria-labelledby="phraseVideoTitle"><h3 id="phraseVideoTitle">구절만 영상 · 비교용</h3><p>이 영상은 따로 재생할 수 있습니다. 두 영상의 자동 동기화나 같은 픽셀 비교를 제공하지 않습니다.</p>
<div class="videoBox"><video id="phraseVideo" controls muted playsinline preload="metadata" aria-label="구절만 무음 영상"></video></div>
<p class="small" id="phraseDisplayNote"></p><p class="notice">원래 입술 영상이 작아 확대해도 세부 모양이 흐릴 수 있습니다. 움직임이 이어지는지 분명하지 않으면 그대로 알려주세요.</p>
</section></div>
<details class="audioSection"><summary>확인한 음성 다시 듣기</summary><div class="audioGrid"><div><label for="phraseAudio">구절 음성</label><audio id="phraseAudio" controls preload="none"></audio></div><div><label for="contextAudio">앞뒤 0.20초 문맥 음성</label><audio id="contextAudio" controls preload="none"></audio></div></div><p class="small">음성은 별도로 재생합니다. 위 초록 표시는 동일 음성의 원본 시간 범위를 영상 시각으로 옮긴 것입니다.</p></details>
<p class="footerNote">이 화면은 판정을 저장하지 않습니다. 구분이 어려우면 채팅에 “여전히 구분하기 어려움”이라고 알려주세요. 한 번에 8개를 모두 판단할 필요는 없습니다.</p>
<p id="error" class="error" role="status" aria-live="polite"></p>
</main><script id="reviewData" type="application/json">__DATA__</script><script>
'use strict';
const rows=JSON.parse(document.getElementById('reviewData').textContent);
const $=id=>document.getElementById(id);
const contextVideo=$('contextVideo'),phraseVideo=$('phraseVideo'),slider=$('frameSlider');
let current=rows[0],selectedIndex=0,stopAt=null,seekSerial=0,lastRequestedFrame=0,callbackId=null;
function number(x){return x.seconds;}
function report(error){$('error').textContent='재생·탐색을 확인하지 못했습니다. 잠시 후 다시 눌러 주세요.';console.error(error);}
function duration(){return number(current.markers.context_display_duration);}
function startTime(){return number(current.markers.phrase_start_in_context_video);}
function endTime(){return number(current.markers.phrase_end_in_context_video);}
function frameFor(t){return Math.max(0,Math.min(current.context_frames-1,Math.floor((t+1e-8)*25)));}
function setRate(){const rate=Number($('speed').value);contextVideo.defaultPlaybackRate=rate;contextVideo.playbackRate=rate;phraseVideo.defaultPlaybackRate=rate;phraseVideo.playbackRate=rate;}
function renderPosition(){const t=contextVideo.currentTime||0;const frame=frameFor(t);slider.value=String(frame);if(!contextVideo.seeking)lastRequestedFrame=frame;
const region=t<startTime()?'구절 앞':t<endTime()?'구절 범위':'구절 뒤';$('region').textContent=region;$('region').classList.toggle('inside',region==='구절 범위');
$('positionTime').textContent=`${t.toFixed(3)} / ${duration().toFixed(3)}초`;$('frameInfo').textContent=`표시 프레임 ${frame+1} / ${current.context_frames}${contextVideo.seeking?' · 탐색 중':''}`;
$('playContext').textContent=contextVideo.paused?'문맥 재생':'일시 정지';}
function ready(video){if(video.readyState>=1)return Promise.resolve();return new Promise((resolve,reject)=>{const ok=()=>{cleanup();resolve();},bad=()=>{cleanup();reject(Error('Video metadata unavailable'));},cleanup=()=>{video.removeEventListener('loadedmetadata',ok);video.removeEventListener('error',bad);};video.addEventListener('loadedmetadata',ok,{once:true});video.addEventListener('error',bad,{once:true});});}
async function seekFrame(index){const generation=++seekSerial;const active=current.review_id;stopAt=null;contextVideo.pause();lastRequestedFrame=Math.max(0,Math.min(current.context_frames-1,index));const target=(lastRequestedFrame+0.5)/25;await ready(contextVideo);if(generation!==seekSerial||active!==current.review_id)return false;
if(Math.abs(contextVideo.currentTime-target)<1e-5&&!contextVideo.seeking){renderPosition();return true;}
return new Promise((resolve,reject)=>{const done=()=>{cleanup();renderPosition();resolve(generation===seekSerial&&active===current.review_id);},bad=()=>{cleanup();reject(Error('Video seek failed'));},cleanup=()=>{contextVideo.removeEventListener('seeked',done);contextVideo.removeEventListener('error',bad);};contextVideo.addEventListener('seeked',done,{once:true});contextVideo.addEventListener('error',bad,{once:true});contextVideo.currentTime=target;renderPosition();});}
async function showPart(edge){const point=edge==='start'?startTime():endTime();const first=Math.max(0,point-0.18);const end=Math.min(duration(),point+0.18);if(await seekFrame(frameFor(first))){stopAt=end;setRate();await contextVideo.play();}}
function playbackTick(){if(stopAt!==null&&!contextVideo.paused&&contextVideo.currentTime>=stopAt){stopAt=null;contextVideo.pause();}renderPosition();if(!contextVideo.paused)callbackId=requestAnimationFrame(playbackTick);else callbackId=null;}
function beginTicks(){if(callbackId===null)callbackId=requestAnimationFrame(playbackTick);}
async function choose(index){++seekSerial;stopAt=null;for(const v of [contextVideo,phraseVideo,$('phraseAudio'),$('contextAudio')])v.pause();selectedIndex=index;current=rows[index];lastRequestedFrame=0;$('error').textContent='';$('qid').textContent=current.review_id;$('phraseTitle').textContent=current.text;
contextVideo.src=current.urls.phrase_context_video;phraseVideo.src=current.urls.phrase_video;$('phraseAudio').src=current.urls.phrase_audio;$('contextAudio').src=current.urls.context_audio;contextVideo.load();phraseVideo.load();
slider.max=String(current.context_frames-1);slider.value='0';const start=100*startTime()/duration(),end=100*endTime()/duration();$('audioRange').style.left=`${start}%`;$('audioRange').style.width=`${end-start}%`;$('startMarker').style.left=`${start}%`;$('endMarker').style.left=`${end}%`;
$('markerTimes').textContent=`확인한 음성 범위: 문맥영상 ${startTime().toFixed(6)}초 ~ ${endTime().toFixed(6)}초 (화면 표시는 반올림)`;
$('phraseDisplayNote').textContent=`구절 영상에는 음성 범위 앞 약 ${(number(current.markers.phrase_display_before_audio)*1000).toFixed(1)}ms·뒤 약 ${(number(current.markers.phrase_display_after_audio)*1000).toFixed(1)}ms의 표시 여유가 있습니다.`;setRate();renderPosition();await seekFrame(0);}
rows.forEach((r,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=`${r.review_id} · ${r.text}`;$('item').append(option);});
$('item').addEventListener('change',()=>choose(Number($('item').value)).catch(report));$('speed').addEventListener('change',setRate);
$('playContext').addEventListener('click',async()=>{try{stopAt=null;if(contextVideo.paused){await ready(contextVideo);if(contextVideo.ended)await seekFrame(0);setRate();await contextVideo.play();}else contextVideo.pause();}catch(e){report(e);}});
$('startPart').addEventListener('click',()=>showPart('start').catch(report));$('endPart').addEventListener('click',()=>showPart('end').catch(report));
$('prevFrame').addEventListener('click',()=>seekFrame(lastRequestedFrame-1).catch(report));$('nextFrame').addEventListener('click',()=>seekFrame(lastRequestedFrame+1).catch(report));
$('jumpStart').addEventListener('click',()=>seekFrame(frameFor(startTime())).catch(report));$('jumpEnd').addEventListener('click',()=>seekFrame(frameFor(endTime())).catch(report));slider.addEventListener('input',()=>seekFrame(Number(slider.value)).catch(report));
contextVideo.addEventListener('play',beginTicks);contextVideo.addEventListener('pause',()=>{stopAt=null;renderPosition();});contextVideo.addEventListener('seeked',renderPosition);contextVideo.addEventListener('timeupdate',renderPosition);contextVideo.addEventListener('ended',renderPosition);
contextVideo.addEventListener('loadedmetadata',setRate);phraseVideo.addEventListener('loadedmetadata',setRate);contextVideo.addEventListener('error',()=>report(Error('Context video failed')));phraseVideo.addEventListener('error',()=>report(Error('Phrase video failed')));
choose(0).catch(report);
</script></body></html>'''


def run() -> None:
    """Create only a new page/manifest with exact markers from bound records."""
    require(not OUTPUT.exists(),"The immutable r5 UI already exists")
    previous,question,evidence,summary=read(PREVIOUS_UI),read(QUESTION),read(EVIDENCE),read(SUMMARY)
    require(evidence["answer"]=="구분하기 어려움" and evidence["phrase_visual_boundary_approved"] is False,"Expected actual inconclusive answer")
    require(evidence["question_manifest"]==bind(QUESTION) and len(question["records"])==8,"Question scope changed")
    require(summary["selected_phrase_visual_confirmed"]==0,"Visual confirmation must remain zero")
    question_rows={r["review_id"]:r for r in question["records"]}
    catalog={}
    frozen_paths=[OLD/"results/completion_manifest.json",BASE/"results/review_snapshot_manifest_r4.json"]
    for path in frozen_paths:
        for item in read(path)["artifacts"]: catalog[str(Path(item["path"]).resolve())]=item
    sources=[bind(p) for p in (PREVIOUS_UI,QUESTION,EVIDENCE,SUMMARY,*frozen_paths)]
    rows=[]
    require([r["review_id"] for r in previous["records"]]==[f"Q{i:02d}" for i in range(1,9)],"Expected fixed eight-item order")
    for old in previous["records"]:
        q=question_rows[old["review_id"]]
        require(old["text"]==q["text"] and old["media"]["phrase_audio"]==q["audio"] and old["media"]["phrase_video"]==q["phrase_video"] and old["media"]["phrase_context_video"]==q["context_video"],"r4 and question media differ")
        for spec in old["media"].values(): checked(spec)
        record_path=Path(old["media"]["phrase_video"]["path"]).parent/"record.json"
        key=str(record_path.resolve());require(key in catalog,"Visual clock source not frozen")
        checked(catalog[key]);sources.append(catalog[key]);record=read(record_path)
        require(record["review_id"]==old["review_id"],"Visual source identity differs")
        videos=record.get("videos",record.get("muted_video"))
        phrase=videos["phrase"];context=videos["context"] if "context" in videos else videos["phrase_context"]
        require({k:phrase[k] for k in ("path","sha256","bytes")}==old["media"]["phrase_video"] and {k:context[k] for k in ("path","sha256","bytes")}==old["media"]["phrase_context_video"],"Clock record does not bind displayed video")
        pw,cw=phrase["clock_window"],context["clock_window"]
        start,end=fraction(pw["semantic_video_start"]),fraction(pw["semantic_video_end"])
        origin,stop=fraction(cw["preview_display_video_start"]),fraction(cw["preview_display_video_end"])
        p_origin,p_stop=fraction(pw["preview_display_video_start"]),fraction(pw["preview_display_video_end"])
        require(origin<start<end<stop and stop-origin==Fraction(context["frames"],25),"Confirmed audio marker falls outside context preview")
        require(fraction(cw["semantic_video_start"])==start-Fraction(1,5) and fraction(cw["semantic_video_end"])==end+Fraction(1,5),"Context is not exactly+-0.20s")
        require(p_origin<=start<end<=p_stop and p_stop-p_origin==Fraction(phrase["frames"],25),"Phrase display clock invalid")
        markers={"meaning":"Confirmed audio interval mapped onto the context preview; not verified visual articulation boundaries.","phrase_start_in_context_video":rational(start-origin),"phrase_end_in_context_video":rational(end-origin),
                 "context_display_duration":rational(stop-origin),"context_source_video_origin":rational(origin),"confirmed_audio_mapped_video_start":rational(start),"confirmed_audio_mapped_video_end":rational(end),
                 "phrase_display_before_audio":rational(start-p_origin),"phrase_display_after_audio":rational(p_stop-end),"offset_application":"Already included once in bound video-clock fields; UI adds no offset."}
        rows.append({**old,"state":"음성 확인 완료 · 입술 경계 구분 어려움","phrase_visual_confirmed":False,"human_visual_status":"inconclusive_group_review","human_visual_answer":"구분하기 어려움",
                     "visual_record":catalog[key],"markers":markers,"context_frames":context["frames"],"phrase_frames":phrase["frames"],"fps":25,"training_ready":False})
    OUTPUT.mkdir(parents=True)
    page=PAGE.replace('__DATA__',json.dumps(rows,ensure_ascii=False,separators=(',',':')).replace('<','\u003c'))
    with (OUTPUT/'index.html').open('x',encoding='utf-8',newline='\n') as stream: stream.write(page)
    routes={route:spec for route,spec in previous['routes'].items() if route!='/'}
    routes['/']=bind(OUTPUT/'index.html')
    for route,spec in routes.items(): checked(spec)
    for spec in sources: checked(spec)
    manifest={'schema':'offline-stage5g-current-review-ui-v1','version':'r5','purpose':'one_item_at_a_time_visual_boundary_aid','summary':bind(SUMMARY),'builder':bind(Path(__file__)),
              'source_bindings':sources,'actual_visual_evidence':bind(EVIDENCE),'previous_ui':bind(PREVIOUS_UI),'records':rows,'routes':routes,'range_handler':previous['range_handler'],
              'current_url':'http://127.0.0.1:8775/','default_review_id':'Q01','default_playback_rate':.5,'current_pointer_modified':False,'only_one_item_visible':True,
              'display_frame_navigation':'seek to (index+0.5)/25 in one context preview; browser precision is limited','marker_semantics':'confirmed_audio_only_not_visual_boundary_truth',
              'playback_coordination':'No global play/pause listener and no two-video synchronization; switching item pauses only the four previously displayed media controls.',
              'new_media_files':0,'new_raw_payload_bytes':0,'new_neural_calls':0,'new_training_updates':0,'new_visual_approvals':0,'approval_persistence':False,'training_ready':False}
    write(OUTPUT/'manifest.json',manifest)
    print(json.dumps({'page':str(OUTPUT/'index.html'),'manifest':bind(OUTPUT/'manifest.json'),'records':len(rows),'routes':len(routes),'new_media_files':0,'pointer_modified':False}),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run()
