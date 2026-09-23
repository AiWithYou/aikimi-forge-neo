import test from 'node:test';
import assert from 'node:assert/strict';
import {tokenFromHash, roundTripDifferences, validateAvailableNodes, importEditable} from '../../extensions-builtin/minimax-h3-studio/comfyui_nodes/Aikimi-H3-WorkflowBridge/web/handoff-core.js';

const copy = x => JSON.parse(JSON.stringify(x));
const prompt = {
 '1': {class_type:'RandomNoise', inputs:{noise_seed:731}},
 '2': {class_type:'Condition', inputs:{prompt:'雨の街', 'ref_images.ref_image_0':['1',0]}},
 '3': {class_type:'BlockSparseAttention',inputs:{selection:'Sol-Attn (adaptive tau)', 'selection.tau':1.3}},
};
const schemas = {RandomNoise:{},Condition:{},BlockSparseAttention:{}};
const record = {format:'aikimi-h3-handoff-v1',token:'a'.repeat(32),metadata:{seed:731},prompt};
function mockApp(transform = x=>x) {
 const calls=[];
 const nodes=[{id:1,widgets:[{name:'noise_seed',value:731},{name:'control_after_generate',value:'randomize'}]}];
 const previous={version:0.4,nodes:[{id:99,type:'UserDraft'}]};
 const app={
  rootGraph:{nodes,serialize:()=>copy(previous)},
  async loadApiJson(value,name){calls.push(['load',copy(value),name]); await Promise.resolve(); this.output=transform(value);},
  async graphToPrompt(){calls.push(['serialize']); return {output:copy(this.output),workflow:{version:0.4,nodes:copy(nodes)}};},
  async loadGraphData(value){calls.push(['restore',copy(value)]);},
  queuePrompt(){throw new Error('MUST NOT GENERATE');},
 };
 return {app,calls,nodes,previous};
}

test('only exact opaque fragment tokens accepted',()=>{
 assert.equal(tokenFromHash('#aikimi-h3='+'a'.repeat(32)), 'a'.repeat(32));
 for(const hash of ['#aikimi-h3=../etc','#aikimi-h3='+'a'.repeat(32)+'&x=1','#other=123']) assert.equal(tokenFromHash(hash),null);
});
test('native asynchronous editor import yields real workflow schema and fixed seed',async()=>{
 const {app,calls,nodes}=mockApp();
 const workflow=await importEditable(app,copy(record),schemas);
 assert.equal(workflow.version,0.4); assert.ok(Array.isArray(workflow.nodes));
 assert.equal(nodes[0].widgets[1].value,'fixed');
 assert.deepEqual(calls.map(c=>c[0]),['load','serialize']);
 assert.deepEqual(calls[0][1],prompt);
 assert.equal(workflow.extra.aikimi_h3.metadata.seed,731);
});
test('native importer cannot mutate source snapshot',async()=>{
 const {app}=mockApp(x=>{x['2']._meta={title:'new'};return x;});
 const snapshot=copy(record); await importEditable(app,snapshot,schemas);
 assert.deepEqual(snapshot,record);
});
test('flattened DynamicCombo values and dynamic references survive',()=>{
 assert.deepEqual(roundTripDifferences(prompt,copy(prompt),schemas),[]);
});
test('dropped dynamic inputs are rejected and previous user canvas restored',async()=>{
 const {app,calls,previous}=mockApp(x=>{delete x['3'].inputs['selection.tau'];return x;});
 await assert.rejects(importEditable(app,record,schemas),/selection.tau/);
 assert.deepEqual(calls.at(-1),['restore',previous]);
});
test('changed seed is rejected and restored',async()=>{
 const {app,calls}=mockApp(x=>{x['1'].inputs.noise_seed=99;return x;});
 await assert.rejects(importEditable(app,record,schemas),/noise_seed/);
 assert.equal(calls.at(-1)[0],'restore');
});
test('missing node fails before touching user canvas',async()=>{
 const {app,calls}=mockApp();
 await assert.rejects(importEditable(app,record,{RandomNoise:{}}),/必要なノード/);
 assert.deepEqual(calls,[]);
});
test('missing model fails before import',()=>{
 const p={'1':{class_type:'UNETLoader',inputs:{unet_name:'missing.safetensors'}}};
 assert.throws(()=>validateAvailableNodes(p,{UNETLoader:{input:{required:{unet_name:[['available.safetensors']]}}}}),/モデル/);
});
test('V1 and V3 model selectors accepted only when exact filename is available',()=>{
 const p={'1':{class_type:'VAELoader',inputs:{vae_name:'h3.safetensors'}}};
 for(const spec of [[['h3.safetensors']],['COMBO',{options:['h3.safetensors']}]]) {
  assert.doesNotThrow(()=>validateAvailableNodes(p,{VAELoader:{input:{required:{vae_name:spec}}}}));
 }
});
test('only verified native default additions allowed',()=>{
 const actual=copy(prompt); actual['1'].inputs.extra=1;
 assert.equal(roundTripDifferences(prompt,actual,schemas).length,1);
 const safe={...schemas,RandomNoise:{input:{optional:{extra:['INT',{default:1}]}}}};
 assert.deepEqual(roundTripDifferences(prompt,actual,safe),[]);
});
test('SaveVideo accepts only an identical flattened codec duplicate',()=>{
 const source={'14':{class_type:'SaveVideo',inputs:{format:'auto',codec:'auto',video:['13',0]}}};
 const schema={SaveVideo:{input:{required:{format:['COMFY_DYNAMICCOMBO_V3']},optional:{codec:['COMFY_DYNAMICCOMBO_V3']}}}};
 const imported=copy(source); imported['14'].inputs['format.codec']='auto';
 assert.deepEqual(roundTripDifferences(source,imported,schema),[]);
 imported['14'].inputs['format.codec']='h264';
 assert.match(roundTripDifferences(source,imported,schema)[0],/format.codec/);
});
test('dropped nodes, connections and injected nodes rejected',()=>{
 const altered=copy(prompt); delete altered['2']; altered['9']={class_type:'Injected',inputs:{}};
 assert.equal(roundTripDifferences(prompt,altered,schemas).length,2);
 const changed=copy(prompt); changed['2'].inputs['ref_images.ref_image_0']=['1',1];
 assert.equal(roundTripDifferences(prompt,changed,schemas).length,1);
});
test('failed restoration disables imported graph and provides old workflow backup',async()=>{
 const {app,nodes,previous}=mockApp(x=>{delete x['2'];return x;});
 app.loadGraphData=async()=>{throw new Error('restore failed');};
 let failure; try{await importEditable(app,record,schemas);}catch(e){failure=e;}
 assert.ok(failure); assert.deepEqual(failure.previousWorkflow,previous); assert.equal(nodes[0].mode,2);
});
test('unsafe integer seed and old frontend explicitly rejected',async()=>{
 const {app,calls}=mockApp();
 await assert.rejects(importEditable(app,{...record,metadata:{seed:2**53}},schemas),/Seed/);
 assert.deepEqual(calls,[]);
 delete app.loadApiJson;
 await assert.rejects(importEditable(app,record,schemas),/フロントエンド/);
});
test('UI conversion failure never reports an API object as a UI workflow',async()=>{
 const {app,calls}=mockApp(); app.graphToPrompt=async()=>({output:copy(prompt),workflow:copy(prompt)});
 await assert.rejects(importEditable(app,record,schemas),/UIワークフロー/);
 assert.equal(calls.at(-1)[0],'restore');
});
