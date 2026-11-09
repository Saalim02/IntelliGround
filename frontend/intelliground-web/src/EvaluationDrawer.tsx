import { useEffect, useState } from 'react'

type Evidence={page_number?:number;snippet?:string;image_path?:string}
type Evaluation={source_relevance?:number;grounding?:number;answer_correctness?:number;citation_quality?:number;overall_confidence?:number;decision?:string;reasons?:string[];claims?:{supported:boolean}[]}
type Message={role:'user'|'assistant';text:string;evidence?:Evidence[];evaluation?:Evaluation|null;answerId?:number;feedback?:boolean|null}

export default function EvaluationDrawer({messages,selectedIndex,conversationId,api,onSelect,onClose}:{messages:Message[];selectedIndex:number|null;conversationId:string;api:(path:string,init?:RequestInit)=>Promise<any>;onSelect:(index:number)=>void;onClose:()=>void}){
 const message=selectedIndex==null?null:messages[selectedIndex]
 const [status,setStatus]=useState<'idle'|'loading'|'completed'|'unavailable'|'failed'>(message?.evaluation?'completed':message?.evidence?.length?'idle':'unavailable')
 const [evaluation,setEvaluation]=useState<Evaluation|null>(message?.evaluation||null)
 const [feedbackValue,setFeedbackValue]=useState<boolean|null>(message?.feedback??null)
 const [reason,setReason]=useState('')
 useEffect(()=>{setEvaluation(message?.evaluation||null);setFeedbackValue(message?.feedback??null);setStatus(message?.evaluation?'completed':message?.evidence?.length?'idle':'unavailable');setReason(message?.evaluation||message?.evidence?.length?'':'This answer has no eligible PDF context.')},[message?.answerId,message?.evaluation,message?.feedback,message?.evidence])
 const evaluate=async()=>{if(!message?.answerId||!conversationId){setStatus('unavailable');setReason('This answer has no eligible PDF context.');return}setStatus('loading');try{const result=await api('/evaluation/answer',{method:'POST',body:JSON.stringify({conversation_id:conversationId,answer_id:message.answerId})});if(result.evaluation){setEvaluation(result.evaluation);setStatus('completed')}else{setStatus('unavailable');setReason(result.reason||'Evaluation is unavailable for this answer.')}}catch(error){setStatus('failed');setReason(error instanceof Error?error.message:'Evaluation failed. Please try again.')}}
 const feedback=async(helpful:boolean)=>{if(!message?.answerId||!conversationId)return;try{await api('/evaluation/answer/feedback',{method:'POST',body:JSON.stringify({conversation_id:conversationId,answer_id:message.answerId,helpful})});setFeedbackValue(helpful)}catch(error){setReason(error instanceof Error?error.message:'Feedback could not be saved.')}}
 const score=(value?:number)=>value==null?'Not evaluated':value.toFixed(4)
 const metrics:[string,number|undefined][]=[['Faithfulness',undefined],['Grounding',evaluation?.grounding],['Answer relevance',undefined],['Answer correctness',evaluation?.answer_correctness],['Citation accuracy',evaluation?.citation_quality],['Retrieval relevance',evaluation?.source_relevance]]
 const assistants=messages.map((item,index)=>({item,index})).filter(item=>item.item.role==='assistant')
 return (
  <aside className="evaluation-panel evaluation-drawer" aria-label="Answer evaluation">
   <div className="drawer-header"><div><h2>Evaluation</h2><small>Selected assistant answer</small></div><button className="drawer-close" onClick={onClose} aria-label="Close evaluation">×</button></div>
   {assistants.length>1&&<label className="evaluation-answer-picker">Answer<select value={selectedIndex??''} onChange={event=>onSelect(Number(event.target.value))}>{assistants.map(({item,index},number)=><option value={index} key={item.answerId||index}>Answer {number+1} · {item.text.slice(0,38)}{item.text.length>38?'…':''}</option>)}</select></label>}
   <div className="evaluation-content">
    {!message? <div className="no-evaluation"><div className="score-ring empty-ring"><b>—</b><small>No answer<br/>selected</small></div><p>Select an assistant answer to inspect its evaluation.</p></div> : <>
     <div className="evaluation-heading"><span>✦</span><h2>LLM-as-a-Judge</h2></div>
     {status==='loading'&&<div className="evaluation-progress">Evaluation in progress…</div>}
     {(status==='failed'||status==='unavailable')&&<div className="evaluation-error">{reason||'Evaluation unavailable for this answer.'}</div>}
     {status==='idle'&&!evaluation&&<div className="no-evaluation"><p>This answer has not been evaluated yet.</p><button className="developer-action" onClick={()=>void evaluate()} disabled={!message.answerId}>Evaluate this answer</button></div>}
     {evaluation&&<>
      <div className="score-row"><div className="score-ring" style={{'--score':`${(evaluation.overall_confidence||0)*360}deg`} as any}><b>{score(evaluation.overall_confidence)}</b><small>Overall<br/>score</small></div><div><span className={`status-pill ${evaluation.decision==='AUTO_APPROVE'?'good':''}`}>{evaluation.decision||'Completed'}</span><p>{evaluation.reasons?.[0]||'Evaluation completed against the retrieved evidence.'}</p></div></div>
      <h3 className="breakdown-title">Answer quality metrics</h3><div className="metric-list">{metrics.map(([label,value])=><div className={`metric ${value==null?'unavailable':''}`} key={label}><span>{label}</span><div><i style={{width:value==null?'0%':`${value*100}%`}}/></div><b>{score(value)}</b></div>)}</div>
      <div className="judge-box"><h3>Evaluation explanation</h3><p>{evaluation.reasons?.join(' ')||'No additional explanation was returned.'}</p>{evaluation.claims?.some(claim=>!claim.supported)&&<small>Unsupported claims were detected in the evaluation results.</small>}</div>
      <div className="evidence-section"><div className="evidence-title"><h3>Evidence and citations</h3></div><p className="muted">{message.evidence?.length?`PDF evidence available on page(s): ${message.evidence.map(item=>item.page_number).filter(Boolean).join(', ')}`:'No PDF evidence was returned.'}</p>{message.evidence?.length&&<button className="developer-action" onClick={onClose}>Open evidence in chat</button>}</div>
     </>}
     <div className="feedback-buttons"><span>Was this answer helpful?</span><button className={feedbackValue===true?'selected':''} onClick={()=>void feedback(true)}>👍 Helpful</button><button className={feedbackValue===false?'selected':''} onClick={()=>void feedback(false)}>👎 Not helpful</button></div>
    </>}
   </div>
  </aside>
 )
}
