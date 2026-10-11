from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urljoin
from urllib.request import urlopen

import gradio
import httpx
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient

from modules import gradio_frontend_compat
from modules.aikimi_security.gradio_file_guard import install_gradio_file_url_guard


class GradioFrontendCompatibilityTests(unittest.TestCase):
    def test_apptree_pending_updates_keep_newer_values_and_partial_fields(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required to execute the compiled app tree registration")
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.APPTREE_ASSET_NAME
        original = path.read_text(encoding="utf-8")
        served = module.build_patched_apptree_asset().content.decode("utf-8")
        script = (
            "import assert from 'node:assert/strict';\n"
            + "const originalSource = "
            + json.dumps(original)
            + ";\nconst servedSource = "
            + json.dumps(served)
            + ";\n"
            + r"""
            function makeTree(source, family = 'dropdown') {
                const start = source.indexOf('const q=');
                const end = source.lastIndexOf('export{');
                assert.ok(start > 0 && end > start);
                // Run the actual compiled AppTree. Signals and the delayed tick
                // are controlled, without assuming browser/Svelte render timing.
                const pendingTicks = [];
                const factory = new Function('u', `
                    const R=()=>({value:undefined}),U=x=>x.value,W=(x,v)=>{x.value=v};
                    const I=()=>[new Set(),new Set()],A=null;
                    const k=['visible','interactive','label','loading_status','register_component','dispatcher'];
                    const L=(id,interactive)=>interactive??true,f=()=>({component:{},runtime:false});
                    const j=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
                    ${source.slice(start,end)}
                    return Q;
                `);
                const AppTree = factory(() => new Promise(resolve => pendingTicks.push(resolve)));
                const nested = family !== 'dropdown';
                const components = [{id:1,type:family,props:{visible:true,value:'initial',
                    row_count:[0,'fixed'],col_count:[2,'fixed']}}];
                if (nested) components.push({id:2,type:family === 'tabs' ? 'tabitem' : 'dropdown',
                    props:{visible:true,id:'pane',label:'initial-child'}});
                const tree = new AppTree(components,
                    {id:0,children:[{id:1,children:nested ? [{id:2,children:[]}] : []}]},[],
                    {root:'http://localhost:7860',api_prefix:'/gradio_api'},{},x=>x,()=>{});
                return {tree, pendingTicks, async flush() {
                    for (const resolve of pendingTicks.splice(0)) resolve();
                    await Promise.resolve();
                    await Promise.resolve();
                }};
            }
            function mount(tree, state = {}, id = 1) {
                const calls = [];
                tree.register_component(id, patch => {
                    calls.push({...patch});
                    Object.assign(state, patch);
                }, () => state);
                return {state, calls};
            }
            async function overlap(source) {
                const {tree, flush} = makeTree(source);
                await tree.update_state(1,{value:'pending-old',choices:['old'],label:'保留',
                    row_count:[0,'fixed'],col_count:[2,'fixed']},false);
                const mounted = mount(tree);
                const immediate = tree.update_state(1,{value:'new',choices:['new']},false);
                assert.equal(mounted.state.value,'new');
                await immediate;
                await flush();
                return mounted.state;
            }
            // The wheel sequence is a counterexample, not a copied replacement.
            assert.equal((await overlap(originalSource)).value,'pending-old');
            const newest = await overlap(servedSource);
            assert.equal(newest.value,'new');
            assert.deepEqual(newest.choices,['new']);
            assert.equal(newest.label,'保留');
            assert.deepEqual(newest.row_count,[0,'fixed']);
            assert.deepEqual(newest.col_count,[2,'fixed']);
            {
                const {tree, flush} = makeTree(servedSource);
                await tree.update_state(1,{choices:['retained'],label:'pending'},false);
                const mounted = mount(tree);
                await tree.update_state(1,{value:'new-only',row_count:[3,'dynamic']},false);
                await flush();
                assert.deepEqual(mounted.state,{choices:['retained'],label:'pending',
                    value:'new-only',row_count:[3,'dynamic']});
            }
            {
                const {tree, pendingTicks, flush} = makeTree(servedSource);
                const mounted = mount(tree);
                await tree.update_state(1,{value:'ordinary'},false);
                assert.equal(pendingTicks.length,0);
                await flush();
                assert.deepEqual(mounted.calls,[{value:'ordinary'}]);
            }
            {
                const {tree, flush} = makeTree(servedSource);
                await tree.update_state(1,{value:'old',label:'retained'},false);
                const first = mount(tree);
                tree.update_state(1,{value:'intermediate'},false);
                const latest = mount(tree);
                tree.update_state(1,{value:'newest'},false);
                await flush();
                assert.deepEqual(first.calls,[{value:'intermediate'}]);
                assert.deepEqual(latest.calls,[{value:'newest'},{value:'newest',label:'retained'}]);
                const remounted = mount(tree);
                await flush();
                assert.deepEqual(remounted.calls,[]);
            }
            {
                const {tree, pendingTicks, flush} = makeTree(servedSource);
                await tree.update_state(1,{loading_status:{status:'pending'}},false);
                const mounted = mount(tree);
                assert.equal(pendingTicks.length,0);
                await flush();
                assert.deepEqual(mounted.calls,[]);
            }
            {
                const {tree, flush} = makeTree(servedSource);
                await tree.update_state(1,{value:'old',loading_status:{status:'pending'}},false);
                const mounted = mount(tree);
                tree.update_state(1,{value:'new',loading_status:{status:'complete'}},false);
                tree.update_state(1,{loading_status:{status:null}},false);
                await flush();
                assert.equal(mounted.state.value,'new');
                assert.deepEqual(mounted.state.loading_status,{status:null});
                assert.equal(Object.hasOwn(mounted.calls.at(-1),'loading_status'),false);
            }
            {
                const {tree, flush} = makeTree(servedSource);
                await tree.update_state(1,{value:'old',label:'trigger'},false);
                const state = {}, calls = [];
                tree.register_component(1, patch => {
                    calls.push({...patch});
                    Object.assign(state,patch);
                    if (patch.label === 'trigger') tree.update_state(1,{value:'reentrant'},false);
                }, () => state);
                tree.update_state(1,{value:'before-flush'},false);
                await flush();
                assert.equal(state.value,'reentrant');
                assert.equal(calls.length,3);
                mount(tree);
                await flush();
                assert.equal(calls.length,3);
            }
            {
                const {tree, flush} = makeTree(servedSource);
                await tree.update_state(1,{value:'old',row_count:[0,'fixed']},false);
                const mounted = mount(tree);
                const tuple = [3,'dynamic'];
                tree.update_state(1,{value:undefined,row_count:null,col_count:tuple},false);
                await flush();
                assert.equal(Object.hasOwn(mounted.state,'value'),true);
                assert.equal(mounted.state.value,undefined);
                assert.equal(mounted.state.row_count,null);
                assert.equal(mounted.state.col_count,tuple);
            }
            {
                const {tree, flush} = makeTree(servedSource);
                await tree.update_state(1,{value:'old'},false);
                const mounted = mount(tree), value = {headers:['LoRA','強度'],data:[]};
                tree.update_state(1,{value},false);
                await flush();
                assert.equal(mounted.state.value,value);
                assert.equal(mounted.calls.at(-1).value,value);
            }
            async function visibility(source) {
                const {tree, flush} = makeTree(source,'column');
                await tree.update_state(2,{visible:false},false);
                mount(tree);
                const parentUpdate = tree.update_state(1,{visible:true});
                const child = mount(tree,{},2);
                await flush();
                await parentUpdate;
                return child;
            }
            assert.equal((await visibility(originalSource)).state.visible,false);
            const visible = await visibility(servedSource);
            assert.equal(visible.state.visible,true);
            assert.deepEqual(visible.calls,[{visible:true},{visible:true}]);
            async function tabParent(source) {
                const {tree, flush} = makeTree(source,'tabs');
                await tree.update_state(1,{initial_tabs:[{component_id:2,id:'pane',label:'old'}]},false);
                const mounted = mount(tree);
                await tree.update_state(2,{label:'latest-child'},false);
                await flush();
                return {label:mounted.state.initial_tabs[0].label,
                    latestReference:mounted.state.initial_tabs === tree.root.children[0].props.props.initial_tabs};
            }
            // The wheel mutates the old array's child label in place, but still
            // replays the old array reference after the new array was applied.
            assert.deepEqual(await tabParent(originalSource),{label:'latest-child',latestReference:false});
            assert.deepEqual(await tabParent(servedSource),{label:'latest-child',latestReference:true});
            console.log('compiled app tree pending-update cases passed');
        """
        )
        result = subprocess.run(  # noqa: S603 - execute the audited frontend snippet
            [node, "--input-type=module", "-"],
            input=script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_apptree_mounted_updates_survive_actual_component_prop_sync(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required to execute the compiled component property sync")
        module = gradio_frontend_compat
        assets = Path(gradio.__file__).resolve().parent / "templates/frontend/assets"
        original = (assets / module.APPTREE_ASSET_NAME).read_text(encoding="utf-8")
        common = (assets / "i18n-Cp0K4Pzb.js").read_text(encoding="utf-8")
        dropdown = (assets / "Dropdown-BNDKcPry.js").read_text(encoding="utf-8")
        script = (
            "import assert from 'node:assert/strict';\n"
            + "const originalSource = "
            + json.dumps(original)
            + ";\nconst servedSource = "
            + json.dumps(module.build_patched_apptree_asset().content.decode("utf-8"))
            + ";\nconst componentSource = "
            + json.dumps(common)
            + ";\nconst servedComponentSource = "
            + json.dumps(module.build_patched_component_state_asset().content.decode("utf-8"))
            + ";\nconst dropdownSource = "
            + json.dumps(dropdown)
            + ";\n"
            + r"""
            // Evaluate the wheel's AppTree, component state and Dropdown change
            // effect. Only signals/ticks are controlled; no browser scheduling
            // or DOM behavior is inferred from this contract.
            const sharedKeys = JSON.parse(componentSource.match(/As=(\[[^\]]+\]);/)[1]);
            const translatedKeys = JSON.parse(componentSource.match(/xs=(\[[^\]]+\]);/)[1]);
            const changeStart = dropdownSource.indexOf('let p=_(ve(f()));ae(()=>');
            const changeEnd = dropdownSource.indexOf(';var w=', changeStart);
            assert.ok(changeStart > 0 && changeEnd > changeStart);
            function make(source, family = 'dropdown', stateSource = servedComponentSource, formatter = x=>x) {
                const effects = [], ticks = [], events = [];
                const stateStart = stateSource.indexOf('class iu{');
                const stateEnd = stateSource.indexOf('const ou=', stateStart);
                assert.ok(stateStart > 0 && stateEnd > stateStart);
                const signal = value => ({value});
                const read = slot => slot.value;
                const write = (slot, value) => {slot.value = value};
                const treeStart = source.indexOf('const q='), treeEnd = source.lastIndexOf('export{');
                const AppTree = new Function('R','U','W','u','k', `
                    const I=()=>[new Set(),new Set()],A=null;
                    const L=(id,interactive)=>interactive??true,f=()=>({component:{},runtime:false});
                    const j=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
                    ${source.slice(treeStart, treeEnd)}return Q;
                `)(signal,read,write,()=>new Promise(resolve=>ticks.push(resolve)),sharedKeys);
                const State = new Function('pt','L','te','Le','As','xs', `
                    const Lt=value=>value,Z=fn=>fn();
                    const Ps=()=>false,io=value=>value;
                    ${stateSource.slice(stateStart, stateEnd)}return iu;
                `)(signal,read,write,fn=>effects.push(fn),sharedKeys,translatedKeys);
                const nested = family !== 'dropdown';
                const components = [{id:1,type:family,props:{visible:true,interactive:true,value:'bf16',
                    label:'initial',choices:[['BF16','bf16'],['Q4','base_q4_k_m']]}}];
                if (nested) components.push({id:2,type:'dropdown',props:{visible:true,interactive:true,value:'child'}});
                const tree = new AppTree(components,
                    {id:0,children:[{id:1,children:nested ? [{id:2,children:[]}] : []}]},[],
                    {root:'http://localhost:7860',api_prefix:'/gradio_api'},{},formatter,
                    (id,event,value)=>events.push({id,event,value}));
                function mount(id = 1) {
                    const config = id === 1 ? tree.root.children[0].props : tree.root.children[0].children[0].props;
                    const firstEffect = effects.length;
                    const state = new State(config);
                    const sync = effects[firstEffect];
                    assert.equal(typeof sync,'function');
                    const changes = [];
                    const changeEffects = [];
                    new Function('_','ve','f','ae','e','t','l',
                        dropdownSource.slice(changeStart, changeEnd)
                    )(signal,value=>value,()=>state.props.value,fn=>changeEffects.push(fn),read,write,
                        {on_change:value=>changes.push(value)});
                    return {state,config,changes,sync,
                        checkChange(){for(const effect of changeEffects)effect()}};
                }
                return {tree,mount,events,async flush(){
                    for(const resolve of ticks.splice(0))resolve();
                    await Promise.resolve();await Promise.resolve();
                }};
            }
            async function restore(source, stateSource) {
                const fixture = make(source,'dropdown',stateSource), mounted = fixture.mount();
                mounted.sync();
                await fixture.tree.update_state(1,{value:'base_q4_k_m',interactive:false,label:'updated'},false);
                mounted.checkChange();
                assert.equal(mounted.state.props.value,'base_q4_k_m');
                mounted.sync();mounted.checkChange();
                return mounted;
            }
            const old = await restore(originalSource,componentSource);
            assert.equal(old.state.props.value,'bf16');
            assert.deepEqual(old.changes,['base_q4_k_m','bf16']);
            const current = await restore(servedSource);
            assert.equal(current.state.props.value,'base_q4_k_m');
            assert.equal(current.state.shared.interactive,false);
            assert.equal(current.state.shared.label,'updated');
            assert.deepEqual(current.changes,['base_q4_k_m']);
            {
                const fixture = make(servedSource), mounted = fixture.mount();
                mounted.sync();
                const value = {headers:['LoRA','強度'],data:[]};
                await fixture.tree.update_state(1,{value},false);
                await fixture.tree.update_state(1,{label:'partial'},false);
                mounted.sync();
                assert.equal(mounted.state.props.value,value);
                assert.equal(mounted.state.shared.label,'partial');
                assert.deepEqual(mounted.state.props.choices,[['BF16','bf16'],['Q4','base_q4_k_m']]);
            }
            {
                const fixture = make(servedSource);
                await fixture.tree.update_state(1,{value:'base_q4_k_m'},false);
                await fixture.tree.update_state(1,{value:'base_q4_k_m'},false);
                assert.deepEqual(fixture.events,[{id:1,event:'change',value:null}]);
                const mounted = fixture.mount();
                mounted.sync();await fixture.flush();mounted.checkChange();
                assert.equal(mounted.state.props.value,'base_q4_k_m');
                assert.deepEqual(mounted.changes,[]);
            }
            {
                const fixture = make(servedSource,'column');
                const parent = fixture.mount(), child = fixture.mount(2);
                parent.sync();child.sync();
                await fixture.tree.update_state(2,{visible:false},false);
                child.sync();
                assert.equal(child.state.shared.visible,false);
                const update = fixture.tree.update_state(1,{visible:true});
                await fixture.flush();await update;
                assert.equal(child.state.shared.visible,true);
                assert.equal(fixture.tree.root.children[0].children[0].original_visibility,false);
                child.sync();
                assert.equal(child.state.shared.visible,true);
            }
            {
                const fixture = make(servedSource), mounted = fixture.mount();
                mounted.sync();
                let reentered = false;
                fixture.tree.register_component(1, patch=>{
                    mounted.state.set_data(patch);
                    if(!reentered){reentered=true;fixture.tree.update_state(1,{value:'reentrant'},false)}
                },mounted.state.get_data.bind(mounted.state));
                await fixture.tree.update_state(1,{value:'outer'},false);
                mounted.sync();
                assert.equal(mounted.state.props.value,'reentrant');
            }
            {
                const fixture = make(servedSource), mounted = fixture.mount();
                // A local input can precede the first source effect. Unchanged
                // construction props must preserve it even on that first flush.
                const custom = {token:'local'};
                mounted.state.set_data({value:'base_q4_k_m',interactive:false,custom});
                mounted.sync();mounted.checkChange();
                assert.equal(mounted.state.props.value,'base_q4_k_m');
                assert.equal(mounted.state.shared.interactive,false);
                assert.equal(mounted.state.props.custom,custom);
                mounted.config.shared_props.loading_status = {status:'pending'};
                mounted.sync();mounted.checkChange();
                assert.equal(mounted.state.props.value,'base_q4_k_m');
                assert.equal(mounted.state.shared.interactive,false);
                assert.deepEqual(mounted.changes,['base_q4_k_m']);
                assert.deepEqual(mounted.state.shared.loading_status,{status:'pending'});
                mounted.config.props.value = 'explicit-new';
                mounted.sync();mounted.checkChange();
                assert.equal(mounted.state.props.value,'explicit-new');
                assert.equal(mounted.state.shared.interactive,false);
                // Same-valued backend output still applies through set_data.
                mounted.state.set_data({value:'local-again'});
                await fixture.tree.update_state(1,{value:'explicit-new'},false);
                mounted.sync();
                assert.equal(mounted.state.props.value,'explicit-new');
            }
            {
                const fixture = make(servedSource), mounted = fixture.mount();
                mounted.sync();
                const value = {headers:['LoRA'],data:[]};
                mounted.config.props.value = value;
                mounted.config.props.new_undefined = undefined;
                mounted.config.props.nan = NaN;
                mounted.sync();
                assert.equal(mounted.state.props.value,value);
                assert.equal(Object.hasOwn(mounted.state.props,'new_undefined'),true);
                mounted.state.set_data({nan:'local'});
                mounted.sync();
                assert.equal(mounted.state.props.nan,'local');
                delete mounted.config.props.new_undefined;
                mounted.sync();
                mounted.state.set_data({new_undefined:'local'});
                mounted.config.props.new_undefined = undefined;
                mounted.sync();
                assert.equal(mounted.state.props.new_undefined,undefined);
            }
            {
                const fixture = make(servedSource), mounted = fixture.mount();
                mounted.sync();
                // A registered component can outlive replacement of its node.
                fixture.tree.root.children = [];
                await fixture.tree.update_state(1,{value:'detached'},false);
                assert.equal(mounted.state.props.value,'detached');
            }
            {
                const fixture = make(servedSource);
                await fixture.tree.update_state(1,{value:'unmounted-old'},false);
                const mounted = fixture.mount();
                await fixture.tree.update_state(1,{value:'mounted-new'},false);
                mounted.sync();
                await fixture.flush();mounted.sync();
                assert.equal(mounted.state.props.value,'mounted-new');
            }
            {
                const fixture = make(servedSource,'dropdown',servedComponentSource,
                    value=>value === 'initial' ? 'translated initial' : value);
                const mounted = fixture.mount();
                assert.equal(mounted.state.shared.label,'translated initial');
                mounted.sync();
                assert.equal(mounted.state.shared.label,'translated initial');
                await fixture.tree.update_state(1,{label:'explicit label'},false);
                mounted.sync();
                assert.equal(mounted.state.shared.label,'explicit label');
                mounted.config.shared_props.label = 'source label';
                mounted.sync();
                assert.equal(mounted.state.shared.label,'source label');
            }
            {
                const oldFixture = make(originalSource,'dropdown',componentSource);
                const oldMounted = oldFixture.mount();
                oldMounted.sync();
                oldMounted.state.set_data({value:'base_q4_k_m',interactive:false});
                oldMounted.config.shared_props.id = 2;
                oldMounted.sync();
                assert.equal(oldMounted.state.props.value,'bf16');
                assert.equal(oldMounted.state.shared.interactive,true);
                const fixture = make(servedSource), mounted = fixture.mount();
                mounted.sync();
                mounted.state.set_data({value:'base_q4_k_m',interactive:false});
                mounted.checkChange();
                mounted.config.shared_props.id = 2;
                mounted.sync();mounted.checkChange();
                // The original effect also registers the same wrapper under
                // the new ID. A different component must receive its source
                // fields even when their raw values match the previous ID.
                assert.equal(mounted.state.shared.id,2);
                assert.equal(mounted.state.props.value,'bf16');
                assert.equal(mounted.state.shared.interactive,true);
                assert.equal((await fixture.tree.get_state(2)).value,'bf16');
                assert.deepEqual(mounted.changes,['base_q4_k_m','bf16']);
                mounted.state.set_data({value:'local second',interactive:false});
                mounted.config.shared_props.loading_status = {status:'pending'};
                mounted.sync();
                assert.equal(mounted.state.props.value,'local second');
                assert.equal(mounted.state.shared.interactive,false);
                mounted.config.shared_props.id = 1;
                mounted.sync();
                assert.equal(mounted.state.props.value,'bf16');
                assert.equal(mounted.state.shared.interactive,true);
            }
            console.log('compiled mounted component prop sync cases passed');
        """
        )
        result = subprocess.run(  # noqa: S603 - execute the audited frontend snippet
            [node, "--input-type=module", "-"],
            input=script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_component_state_sync_snippet_and_hash_fail_closed(self):
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.COMPONENT_STATE_ASSET_NAME
        source = path.read_bytes()
        snippet = module._ORIGINAL_COMPONENT_STATE_SYNC
        self.assertEqual(source.count(snippet), 1)
        for name, replacement in (("missing", b""), ("duplicated", snippet * 2)):
            changed = source.replace(snippet, replacement, 1)
            with (
                self.subTest(mutation=name),
                patch.object(Path, "read_bytes", return_value=changed),
                patch.object(module, "COMPONENT_STATE_ORIGINAL_SHA256", hashlib.sha256(changed).hexdigest()),
                self.assertRaisesRegex(module.GradioFrontendCompatibilityError, "snippet was not found exactly once"),
            ):
                module.build_patched_component_state_asset()
        with (
            patch.object(Path, "read_bytes", return_value=source + b" "),
            self.assertRaisesRegex(module.GradioFrontendCompatibilityError, "audited SHA-256"),
        ):
            module.build_patched_component_state_asset()

    def test_component_state_route_is_verified_and_keeps_the_wheel(self):
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.COMPONENT_STATE_ASSET_NAME
        before = path.read_bytes()
        asset = module.build_patched_component_state_asset()
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(hashlib.sha256(before).hexdigest(), asset.original_sha256)
        self.assertEqual(hashlib.sha256(asset.content).hexdigest(), asset.patched_sha256)
        self.assertNotIn(module._ORIGINAL_COMPONENT_STATE_SYNC, asset.content)
        self.assertEqual(asset.content.count(module._CHANGED_COMPONENT_STATE_SYNC), 1)
        app = module.create_gradio_compatibility_app(module.build_patched_tabs_asset())
        client = TestClient(app)
        url = f"/assets/{asset.filename}"
        response = client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, asset.content)
        self.assertEqual(response.headers["etag"], f'"{asset.patched_sha256}"')
        head = client.head(url)
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(head.headers["content-length"], str(len(asset.content)))
        cached = client.get(url, headers={"If-None-Match": f'"{asset.patched_sha256}"'})
        self.assertEqual(cached.status_code, 304)
        self.assertEqual(cached.content, b"")
        with patch.object(gradio, "__version__", "6.29.0"):
            with self.assertRaises(module.GradioFrontendCompatibilityError):
                module.build_patched_component_state_asset()

    def test_apptree_pending_update_snippets_fail_closed_when_missing_or_duplicated(self):
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.APPTREE_ASSET_NAME
        source = path.read_bytes()
        snippets = (
            b"r&&(this.#r.delete(s),u().then(()=>{const n=this.#s.get(s);n&&n(r)}))",
            b"const o=this.#s.get(s);if(o)o&&o(i);else{",
            b"s.children.forEach(e=>{const r=this.#s.get(e.id);r&&r(i),this.update_visibility(e,i)})",
            b"const h=this.#s.get(e.id);h&&h({initial_tabs:e.props.props.initial_tabs})",
            b"prepare(){const[s,i]=I(this.#e);this.#l=s,this.#c=i}",
        )
        for snippet in snippets:
            self.assertEqual(source.count(snippet), 1)
            for name, replacement in (("missing", b""), ("duplicated", snippet * 2)):
                changed = source.replace(snippet, replacement, 1)
                with (
                    self.subTest(snippet=snippet, mutation=name),
                    patch.object(Path, "read_bytes", return_value=changed),
                    patch.object(module, "APPTREE_ORIGINAL_SHA256", hashlib.sha256(changed).hexdigest()),
                    self.assertRaisesRegex(
                        module.GradioFrontendCompatibilityError, "snippet was not found exactly once"
                    ),
                ):
                    module.build_patched_apptree_asset()

    def test_dataframe_row_count_preserves_compiled_empty_row_semantics(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required to execute the compiled dataframe empty-row condition")
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.DATAFRAME_ASSET_NAME
        source = path.read_text(encoding="utf-8")
        asset = module.build_patched_dataframe_asset()
        script = (
            "import assert from 'node:assert/strict';\n"
            + "const originalSource = "
            + json.dumps(source)
            + ";\nconst servedSource = "
            + json.dumps(asset.content.decode("utf-8"))
            + ";\n"
            + r"""
            function between(source, start, end) {
                const offset = source.indexOf(start);
                assert.notEqual(offset, -1, start);
                assert.equal(source.indexOf(start, offset + 1), -1, start);
                const finish = source.indexOf(end, offset + start.length);
                assert.notEqual(finish, -1, end);
                return source.slice(offset + start.length, finish);
            }
            function rowGetter(source) {
                const matches = source.match(/get row_count\(\)\{return r\.props\.row_count[^}]*\}/g) ?? [];
                assert.equal(matches.length, 1);
                return matches[0];
            }
            function columnGetter(source) {
                const matches = source.match(/get col_count\(\)\{return r\.props\.col_count[^}]*\}/g) ?? [];
                assert.equal(matches.length, 1);
                return matches[0];
            }
            const headerArrow = between(originalSource, 'M=R(', '),I=R(()=>r.props.value?.data??[])');
            const headerDerived = '()=>{let f=o()??[];' + between(
                originalSource, 'B=R(()=>{let f=o()??[];', '),ie=R(()=>l(B).map('
            );
            const dataStart = 'M=R(()=>r.props.value?.headers??[]),I=R(';
            const dataArrow = between(originalSource, dataStart, '),b=R(');
            const emptyRowArrow = between(originalSource, 'H(yr,', ')}ae(()=>{wn=');
            assert.equal(dataArrow, '()=>r.props.value?.data??[]');
            assert.equal(dataArrow, between(servedSource, dataStart, '),b=R('));
            assert.equal(emptyRowArrow, 'f=>{i().length===0&&a()&&t.row_count[1]==="dynamic"&&f(br)}');
            assert.equal(emptyRowArrow, between(servedSource, 'H(yr,', ')}ae(()=>{wn='));
            assert.equal(originalSource.split('get values(){return l(I)}').length, 2);
            // Execute the actual compiled lazy getter/condition, without a
            // browser renderer or Svelte scheduling assumptions.
            const R = fn => fn, l = signal => signal();
            const make = source => new Function('r', 'R', 'l',
                `const M=R(${headerArrow}),I=R(${dataArrow});return({get headers(){return l(M)},`
                + `get values(){return l(I)},${columnGetter(source)},${rowGetter(source)}});`
            );
            const original = make(originalSource), guarded = make(servedSource);
            const deriveHeaders = new Function('o', 't', `return (${headerDerived})();`);
            const runCondition = new Function('i', 'a', 't', 'br', 'f', `(${emptyRowArrow})(f);`);
            function canAddEmptyRow(node, editable = true) {
                const br = Symbol('empty-row-renderer');
                let allowed = false;
                runCondition(() => node.values, () => editable, node, br, renderer => {
                    assert.equal(renderer, br);
                    allowed = true;
                });
                return allowed;
            }
            function unchanged(props, run) {
                const value = props.value, headers = value?.headers, data = value?.data;
                const count = props.row_count, hadCount = Object.hasOwn(props, 'row_count');
                const before = JSON.stringify(props);
                run();
                assert.equal(props.value, value);
                assert.equal(props.value?.headers, headers);
                assert.equal(props.value?.data, data);
                assert.equal(props.row_count, count);
                assert.equal(Object.hasOwn(props, 'row_count'), hadCount);
                assert.equal(JSON.stringify(props), before);
            }
            const headers = Object.freeze(['LoRA', '強度']);
            const empty = Object.freeze([]), rows = Object.freeze([Object.freeze(['chosen', 0.5])]);
            let cases = 0;
            function check(run) { run(); cases++; }
            function propsWithoutRows(state) {
                const props = { value: { headers, data: empty } };
                if (state !== 'absent') props.row_count = state === 'null' ? null : undefined;
                return props;
            }
            for (const state of ['absent', 'undefined', 'null']) {
                check(() => {
                    const props = propsWithoutRows(state);
                    assert.throws(() => canAddEmptyRow(original({ props }, R, l)),
                        error => error instanceof TypeError && /reading '1'/.test(error.message));
                    const node = guarded({ props }, R, l);
                    unchanged(props, () => {
                        assert.equal(canAddEmptyRow(node), false);
                        assert.deepEqual(node.row_count, [0, 'fixed']);
                        assert.equal(node.values, empty);
                    });
                });
            }
            for (const tuple of [[0, 'fixed'], [0, 'dynamic'], [3, 'fixed'], [3, 'dynamic']]) {
                check(() => {
                    const row_count = Object.freeze(tuple);
                    const props = { value: { headers, data: empty }, row_count };
                    const node = guarded({ props }, R, l);
                    for (const data of [empty, rows]) {
                        props.value.data = data;
                        unchanged(props, () => {
                            assert.equal(node.row_count, row_count);
                            assert.equal(node.values, data);
                            assert.equal(canAddEmptyRow(node), data.length === 0 && tuple[1] === 'dynamic');
                            assert.equal(canAddEmptyRow(node), canAddEmptyRow(original({ props }, R, l)));
                        });
                    }
                });
            }
            for (const state of ['absent', 'undefined', 'null']) {
                check(() => {
                    const props = propsWithoutRows(state);
                    const node = guarded({ props }, R, l);
                    assert.equal(canAddEmptyRow(node), false);
                    for (const kind of ['dynamic', 'fixed']) {
                        const restored = Object.freeze([0, kind]);
                        props.row_count = restored;
                        unchanged(props, () => {
                            assert.equal(node.row_count, restored);
                            assert.equal(canAddEmptyRow(node), kind === 'dynamic');
                        });
                        props.value.data = rows;
                        unchanged(props, () => assert.equal(canAddEmptyRow(node), false));
                        props.value.data = empty;
                    }
                });
            }
            for (const initial of [empty, undefined, null]) {
                check(() => {
                    const props = { value: { headers, data: initial } };
                    const node = guarded({ props }, R, l);
                    for (const data of [initial, rows, empty, null, undefined]) {
                        props.value.data = data;
                        unchanged(props, () => {
                            assert.deepEqual(node.values, data ?? []);
                            assert.deepEqual(node.row_count, [data?.length ?? 0, 'fixed']);
                            assert.equal(canAddEmptyRow(node), false);
                        });
                    }
                });
            }
            for (const value of [undefined, null]) {
                check(() => {
                    const props = { value };
                    const node = guarded({ props }, R, l);
                    unchanged(props, () => {
                        assert.deepEqual(node.values, []);
                        assert.deepEqual(node.row_count, [0, 'fixed']);
                        assert.equal(canAddEmptyRow(node), false);
                    });
                });
            }
            check(() => {
                const props = propsWithoutRows('absent');
                assert.equal(canAddEmptyRow(original({ props }, R, l), false), false);
                assert.equal(canAddEmptyRow(guarded({ props }, R, l), false), false);
                props.row_count = Object.freeze([0, 'dynamic']);
                assert.equal(canAddEmptyRow(guarded({ props }, R, l), false), false);
            });
            check(() => {
                const props = { value: { headers, data: empty } };
                const node = guarded({ props }, R, l);
                unchanged(props, () => {
                    assert.deepEqual(node.row_count, [0, 'fixed']);
                    assert.deepEqual(node.col_count, [2, 'fixed']);
                    assert.deepEqual(deriveHeaders(() => node.headers, node), headers);
                    assert.equal(canAddEmptyRow(node), false);
                });
                for (const [rowKind, columnKind] of [['dynamic', 'fixed'], ['fixed', 'dynamic']]) {
                    const restoredRows = Object.freeze([0, rowKind]);
                    const restoredColumns = Object.freeze([3, columnKind]);
                    props.row_count = restoredRows;
                    props.col_count = restoredColumns;
                    unchanged(props, () => {
                        assert.equal(node.row_count, restoredRows);
                        assert.equal(node.col_count, restoredColumns);
                        assert.deepEqual(deriveHeaders(() => node.headers, node), columnKind === 'fixed'
                            ? ['LoRA', '強度', '2'] : ['LoRA', '強度']);
                        assert.equal(canAddEmptyRow(node), rowKind === 'dynamic');
                    });
                }
            });
            console.log(`compiled dataframe row_count: ${cases} scenarios passed`);
            """
        )
        result = subprocess.run(  # noqa: S603 - execute expressions extracted from the audited frontend asset
            [node, "--input-type=module", "-"],
            input=script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compiled dataframe row_count: 17 scenarios passed", result.stdout)

    def test_dataframe_row_count_getter_missing_or_duplicated_fails_closed(self):
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.DATAFRAME_ASSET_NAME
        source = path.read_bytes()
        getter = b"get row_count(){return r.props.row_count}"
        self.assertEqual(source.count(getter), 1)
        for name, modified in (
            ("missing", source.replace(getter, b"get row_count(){return undefined}", 1)),
            ("duplicated", source + getter),
        ):
            with (
                self.subTest(snippet=name),
                patch.object(module, "DATAFRAME_ORIGINAL_SHA256", hashlib.sha256(modified).hexdigest()),
                patch.object(Path, "read_bytes", return_value=modified),
            ):
                with self.assertRaisesRegex(
                    module.GradioFrontendCompatibilityError, "row-count getter snippet was not found exactly once"
                ):
                    module.build_patched_dataframe_asset()

    def test_dataframe_column_count_preserves_compiled_header_semantics(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required to execute the compiled dataframe headers")
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.DATAFRAME_ASSET_NAME
        source = path.read_text(encoding="utf-8")
        asset = module.build_patched_dataframe_asset()
        script = (
            "import assert from 'node:assert/strict';\n"
            + "const originalSource = "
            + json.dumps(source)
            + ";\nconst servedSource = "
            + json.dumps(asset.content.decode("utf-8"))
            + ";\n"
            + r"""
            function between(source, start, end) {
                const offset = source.indexOf(start);
                assert.notEqual(offset, -1, start);
                assert.equal(source.indexOf(start, offset + 1), -1, start);
                const finish = source.indexOf(end, offset + start.length);
                assert.notEqual(finish, -1, end);
                return source.slice(offset + start.length, finish);
            }
            function columnGetter(source) {
                const matches = source.match(/get col_count\(\)\{return r\.props\.col_count[^}]*\}/g) ?? [];
                assert.equal(matches.length, 1);
                return matches[0];
            }
            const headerArrow = between(originalSource, 'M=R(', '),I=R(()=>r.props.value?.data??[])');
            const headerDerived = '()=>{let f=o()??[];' + between(
                originalSource, 'B=R(()=>{let f=o()??[];', '),ie=R(()=>l(B).map('
            );
            assert.equal(headerArrow, '()=>r.props.value?.headers??[]');
            assert.equal(headerDerived, '()=>{let f=o()??[];' + between(
                servedSource, 'B=R(()=>{let f=o()??[];', '),ie=R(()=>l(B).map('
            ));
            assert.equal(originalSource.split('get headers(){return l(M)}').length, 2);
            // Evaluate the actual compiled expressions; this models lazy reads,
            // not Svelte's scheduler or a browser mount lifecycle.
            const R = fn => fn, l = signal => signal();
            const make = source => new Function('r', 'R', 'l',
                `const M=R(${headerArrow});return({get headers(){return l(M)},${columnGetter(source)}});`
            );
            const original = make(originalSource), guarded = make(servedSource);
            const derive = new Function('o', 't', `return (${headerDerived})();`);
            const evaluate = node => derive(() => node.headers, node);
            const headers = Object.freeze(['LoRA', '強度']);
            const data = Object.freeze([Object.freeze(['chosen', 0.5])]);
            let cases = 0;
            function check(run) { run(); cases++; }
            for (const state of ['undefined', 'null', 'absent']) {
                check(() => {
                    const value = { headers, data };
                    const props = { value };
                    if (state !== 'absent') props.col_count = state === 'null' ? null : undefined;
                    const before = JSON.stringify(props);
                    const hadCount = Object.hasOwn(props, 'col_count');
                    assert.throws(() => evaluate(original({ props }, R, l)),
                        error => error instanceof TypeError && /reading '1'/.test(error.message));
                    const node = guarded({ props }, R, l);
                    assert.deepEqual(evaluate(node), ['LoRA', '強度']);
                    assert.deepEqual(node.col_count, [2, 'fixed']);
                    assert.equal(props.value, value);
                    assert.equal(props.value.headers, headers);
                    assert.equal(props.value.data, data);
                    assert.equal(Object.hasOwn(props, 'col_count'), hadCount);
                    assert.equal(JSON.stringify(props), before);
                });
            }
            for (const tuple of [[2, 'fixed'], [4, 'fixed'], [4, 'dynamic'], [0, 'fixed']]) {
                check(() => {
                    const col_count = Object.freeze(tuple);
                    const value = { headers, data };
                    const props = { value, col_count };
                    const before = JSON.stringify(props);
                    const node = guarded({ props }, R, l);
                    assert.equal(node.col_count, col_count);
                    assert.deepEqual(evaluate(node), evaluate(original({ props }, R, l)));
                    assert.equal(props.value, value);
                    assert.equal(props.value.headers, headers);
                    assert.equal(props.value.data, data);
                    assert.equal(props.col_count, col_count);
                    assert.equal(JSON.stringify(props), before);
                });
            }
            for (const absent of [undefined, null]) {
                for (const kind of ['fixed', 'dynamic']) {
                    check(() => {
                        const props = { value: { headers, data }, col_count: absent };
                        const node = guarded({ props }, R, l);
                        assert.deepEqual(evaluate(node), ['LoRA', '強度']);
                        const restored = Object.freeze([4, kind]);
                        props.col_count = restored;
                        assert.equal(node.col_count, restored);
                        assert.deepEqual(evaluate(node), kind === 'fixed'
                            ? ['LoRA', '強度', '2', '3'] : ['LoRA', '強度']);
                        assert.equal(props.value.headers, headers);
                        assert.equal(props.value.data, data);
                    });
                }
            }
            for (const initial of [[], undefined, null]) {
                check(() => {
                    const value = { headers: initial, data };
                    const props = { value };
                    const node = guarded({ props }, R, l);
                    assert.deepEqual(evaluate(node), []);
                    for (const next of [headers, Object.freeze(['LoRA', '強度', 'Note']), [], null]) {
                        value.headers = next;
                        const expected = next ?? [];
                        const before = JSON.stringify(props);
                        assert.deepEqual(node.col_count, [expected.length, 'fixed']);
                        assert.deepEqual(evaluate(node), expected);
                        assert.equal(value.headers, next);
                        assert.equal(value.data, data);
                        assert.equal(props.value, value);
                        assert.equal(Object.hasOwn(props, 'col_count'), false);
                        assert.equal(JSON.stringify(props), before);
                    }
                    value.headers = headers;
                    for (const kind of ['fixed', 'dynamic']) {
                        const restored = Object.freeze([3, kind]);
                        props.col_count = restored;
                        assert.equal(node.col_count, restored);
                        assert.deepEqual(evaluate(node), kind === 'fixed'
                            ? ['LoRA', '強度', '2'] : ['LoRA', '強度']);
                    }
                });
            }
            for (const kind of ['fixed', 'dynamic']) {
                check(() => {
                    const empty = Object.freeze([]), col_count = Object.freeze([3, kind]);
                    const props = { value: { headers: empty, data }, col_count };
                    const node = guarded({ props }, R, l);
                    assert.deepEqual(evaluate(node), ['0', '1', '2']);
                    assert.deepEqual(evaluate(node), evaluate(original({ props }, R, l)));
                    assert.equal(node.col_count, col_count);
                    assert.equal(props.value.headers, empty);
                    assert.equal(props.value.data, data);
                });
            }
            for (const value of [undefined, null]) {
                check(() => {
                    const props = { value };
                    assert.deepEqual(evaluate(guarded({ props }, R, l)), []);
                    assert.equal(props.value, value);
                    assert.equal(Object.hasOwn(props, 'col_count'), false);
                });
            }
            console.log(`compiled dataframe col_count: ${cases} scenarios passed`);
            """
        )
        result = subprocess.run(  # noqa: S603 - execute expressions extracted from the audited frontend asset
            [node, "--input-type=module", "-"],
            input=script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compiled dataframe col_count: 18 scenarios passed", result.stdout)

    def test_dataframe_column_count_getter_missing_or_duplicated_fails_closed(self):
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.DATAFRAME_ASSET_NAME
        source = path.read_bytes()
        getter = b"get col_count(){return r.props.col_count}"
        self.assertEqual(source.count(getter), 1)
        for name, modified in (
            ("missing", source.replace(getter, b"get col_count(){return undefined}", 1)),
            ("duplicated", source + getter),
        ):
            with (
                self.subTest(snippet=name),
                patch.object(module, "DATAFRAME_ORIGINAL_SHA256", hashlib.sha256(modified).hexdigest()),
                patch.object(Path, "read_bytes", return_value=modified),
            ):
                with self.assertRaisesRegex(
                    module.GradioFrontendCompatibilityError, "column-count getter snippet was not found exactly once"
                ):
                    module.build_patched_dataframe_asset()

    def test_tab_sync_preserves_registry_when_update_omits_initial_tabs(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required to execute the compiled tab synchronization")
        module = gradio_frontend_compat
        asset = module.build_patched_tabs_asset()
        self.assertIn(module._STATIC_INITIAL_TAB_SYNC, asset.content)
        self.assertIn(module._STATIC_TAB_REGISTRATION, asset.content)
        script = (
            """
            import assert from 'node:assert/strict';
            const current = { id: 'active', label: 'Renamed', visible: false, interactive: false };
            const l = { value: [current] }, he = { value: [current] }, I = { value: [] };
            const le = { value: false }, we = { value: false }, ae = new Set([0]);
            const P = { value: 'active' }, X = { value: 0 };
            const e = state => state.value, d = (state, value) => state.value = value;
            const N = d, v = () => e(P);
            const Ce = () => Promise.resolve();
        """
            + module._STATIC_INITIAL_TAB_SYNC.decode("utf-8")
            + "\nconst registry = ({"
            + module._STATIC_TAB_REGISTRATION.decode("utf-8")
            + "});\n"
            + """
            Se(undefined);
            Se(null);
            await Promise.resolve();
            assert.deepEqual(e(l), [current]);
            assert.deepEqual(e(he), [current]);
            const added = { id: 'new', label: 'New', visible: true, interactive: true };
            Se([{ id: 'active', label: 'Old', visible: true, interactive: true }, added]);
            await Promise.resolve();
            assert.deepEqual(e(l), [current, added]);
            assert.deepEqual(e(he), [current, added]);
            registry.register_tab(current, 0);
            assert.equal(e(P), 'new');
            assert.equal(e(X), 1);
            registry.register_tab({ ...added, visible: 'hidden' }, 1);
            assert.equal(e(P), false);
            assert.equal(e(X), -1);
            registry.register_tab({ ...current, visible: true, interactive: false }, 0);
            assert.equal(e(P), false);
            registry.register_tab(added, 1);
            assert.equal(e(P), 'new');
            Se([]);
            await Promise.resolve();
            assert.deepEqual(e(l), []);
            assert.deepEqual(e(he), []);
        """
        )
        result = subprocess.run(  # noqa: S603 - execute the audited frontend snippet
            [node, "--input-type=module", "-"],
            input=script,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_apptree_visibility_patch_is_exact_served_and_keeps_the_wheel(self):
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.APPTREE_ASSET_NAME
        before = path.read_bytes()
        asset = module.build_patched_apptree_asset()
        self.assertEqual(before, path.read_bytes())
        self.assertNotIn(module._ORIGINAL_APPTREE_VISIBILITY_SYNC, asset.content)
        self.assertIn(module._REACTIVE_APPTREE_VISIBILITY_SYNC, asset.content)
        app = module.create_gradio_compatibility_app(module.build_patched_tabs_asset())
        response = TestClient(app).get(f"/assets/{asset.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, asset.content)
        self.assertEqual(response.headers["etag"], f'"{asset.patched_sha256}"')
        for attribute, value in (
            ("APPTREE_ORIGINAL_SHA256", "0" * 64),
            ("_ORIGINAL_APPTREE_VISIBILITY_SYNC", b"missing audited snippet"),
        ):
            with self.subTest(attribute=attribute), patch.object(module, attribute, value):
                with self.assertRaises(module.GradioFrontendCompatibilityError):
                    module.build_patched_apptree_asset()
        with patch.object(gradio, "__version__", "6.29.0"):
            with self.assertRaises(module.GradioFrontendCompatibilityError):
                module.build_patched_apptree_asset()

    def test_dataframe_route_is_verified_and_does_not_modify_the_wheel(self):
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.DATAFRAME_ASSET_NAME
        before = path.read_bytes()
        asset = module.build_patched_dataframe_asset()
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(hashlib.sha256(before).hexdigest(), asset.original_sha256)
        self.assertEqual(hashlib.sha256(asset.content).hexdigest(), asset.patched_sha256)
        self.assertEqual(before.count(b"get col_count(){return r.props.col_count}"), 1)
        self.assertEqual(asset.content.count(b"get col_count(){return r.props.col_count}"), 0)
        self.assertEqual(asset.content.count(b'get col_count(){return r.props.col_count??[l(M).length,"fixed"]}'), 1)
        self.assertEqual(before.count(b"get row_count(){return r.props.row_count}"), 1)
        self.assertEqual(asset.content.count(b"get row_count(){return r.props.row_count}"), 0)
        self.assertEqual(asset.content.count(b'get row_count(){return r.props.row_count??[l(I).length,"fixed"]}'), 1)
        self.assertNotIn(module._ORIGINAL_TABLE_RESIZE, asset.content)
        self.assertEqual(asset.content.count(module._REACTIVE_TABLE_RESIZE), 1)
        app = module.create_gradio_compatibility_app(module.build_patched_tabs_asset())
        client = TestClient(app)
        response = client.get(f"/assets/{asset.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, asset.content)
        self.assertNotEqual(response.content, before)
        self.assertEqual(response.headers["etag"], f'"{asset.patched_sha256}"')
        head = client.head(f"/assets/{asset.filename}")
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(head.headers["content-length"], str(len(asset.content)))
        self.assertEqual(head.headers["etag"], f'"{asset.patched_sha256}"')
        cached = client.get(f"/assets/{asset.filename}", headers={"If-None-Match": f'"{asset.patched_sha256}"'})
        self.assertEqual(cached.status_code, 304)
        self.assertEqual(cached.content, b"")
        self.assertEqual(cached.headers["etag"], f'"{asset.patched_sha256}"')
        with patch.object(module, "DATAFRAME_ORIGINAL_SHA256", "0" * 64):
            with self.assertRaises(module.GradioFrontendCompatibilityError):
                module.build_patched_dataframe_asset()

    def test_webui_validates_patch_before_starting_gradio_listener(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "webui.py").read_text(encoding="utf-8")

        self.assertLess(
            source.index("patched_tabs = build_patched_tabs_asset()"),
            source.index("app, local_url, share_url = shared.demo.launch("),
        )

    def test_exact_pinned_asset_is_patched_without_modifying_site_packages(self):
        asset_path = (
            Path(gradio.__file__).resolve().parent
            / "templates"
            / "frontend"
            / "assets"
            / gradio_frontend_compat.TABS_ASSET_NAME
        )
        before = asset_path.read_bytes()

        patched = gradio_frontend_compat.build_patched_tabs_asset()

        self.assertEqual(hashlib.sha256(before).hexdigest(), patched.original_sha256)
        self.assertEqual(before, asset_path.read_bytes())
        self.assertNotEqual(before, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_OVERFLOW_FUNCTION, patched.content)
        self.assertIn(gradio_frontend_compat._NO_OVERFLOW_FUNCTION, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_INITIAL_TAB_SYNC, patched.content)
        self.assertIn(gradio_frontend_compat._STATIC_INITIAL_TAB_SYNC, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_TAB_REGISTRATION, patched.content)
        self.assertIn(gradio_frontend_compat._STATIC_TAB_REGISTRATION, patched.content)
        self.assertEqual(hashlib.sha256(patched.content).hexdigest(), patched.patched_sha256)
        self.assertEqual(patched.patched_sha256, gradio_frontend_compat.PATCHED_ASSET_SHA256)

    def test_preconfigured_route_is_available_during_gradio_auto_launch(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]

        asset = gradio_frontend_compat.build_patched_tabs_asset()
        prepared_app = gradio_frontend_compat.create_gradio_compatibility_app(asset)
        self.assertIs(prepared_app.router.default_response_class, ORJSONResponse)
        install_gradio_file_url_guard(prepared_app)
        fetched_during_open: list[bytes] = []
        external_status_during_open: list[tuple[int, str | None]] = []

        def fetch_from_browser_open(local_url: str) -> bool:
            with urlopen(urljoin(local_url, f"assets/{asset.filename}"), timeout=5) as response:  # noqa: S310
                fetched_during_open.append(response.read())
            external = httpx.get(
                local_url.rstrip("/") + "/gradio_api/file=https://redirect.invalid/pre-listener",
                follow_redirects=False,
                timeout=5,
            )
            external_status_during_open.append((external.status_code, external.headers.get("location")))
            return True

        with gradio.Blocks(analytics_enabled=False) as demo:
            gradio.Markdown("pre-listener compatibility route")

        try:
            with patch("webbrowser.open", side_effect=fetch_from_browser_open):
                app, _, _ = demo.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    prevent_thread_lock=True,
                    quiet=True,
                    inbrowser=True,
                    _app=prepared_app,
                )
        finally:
            demo.close()

        self.assertIs(app, prepared_app)
        self.assertEqual(fetched_during_open, [asset.content])
        self.assertEqual(external_status_during_open, [(403, None)])

    def test_webui_opens_browser_only_after_routes_and_middleware_are_ready(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "webui.py").read_text(encoding="utf-8")

        self.assertIn("inbrowser=False", source)
        self.assertIn("debug=False", source)
        self.assertIn("show_error=cmd_opts.gradio_debug", source)
        self.assertIn("ssr_mode=False", source)
        self.assertIn('os.environ["GRADIO_DEBUG"] = "0"', source)
        self.assertIn('os.environ["GRADIO_DEBUG"] = previous_gradio_debug', source)
        self.assertIn("tunnel_baseline = gradio_runtime.tunnel_snapshot()", source)
        self.assertNotIn("shared.demo.close()", source)
        self.assertGreaterEqual(source.count("gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)"), 5)
        launch_block = source.split("app, local_url, share_url = shared.demo.launch(", 1)[1].split(
            'startup_timer.record("gradio launch")',
            1,
        )[0]
        self.assertIn("except Exception:", launch_block)
        self.assertIn("gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)", launch_block)
        self.assertLess(
            source.index("install_gradio_file_url_guard(prepared_app)"), source.index("shared.demo.launch(")
        )
        self.assertLess(source.index("ui.setup_ui_api(app)"), source.index("webbrowser.open(browser_url)"))
        self.assertLess(
            source.index('with startup_timer.subcategory("app_started_callback")'),
            source.index("webbrowser.open(browser_url)"),
        )
        post_launch_block = source.split('startup_timer.record("gradio launch")', 1)[1].split(
            "server_command = shared.state.wait_for_server_command",
            1,
        )[0]
        self.assertIn("initialize_util.setup_middleware(app)", post_launch_block)
        self.assertIn("ui.setup_ui_api(app)", post_launch_block)
        self.assertIn("webbrowser.open(browser_url)", post_launch_block)
        self.assertIn("except Exception:", post_launch_block)
        self.assertIn("gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)", post_launch_block)

    def test_version_mismatch_fails_closed(self):
        with patch.object(gradio, "__version__", "6.18.0"):
            with self.assertRaisesRegex(
                gradio_frontend_compat.GradioFrontendCompatibilityError,
                "not supported",
            ):
                gradio_frontend_compat.build_patched_tabs_asset()

    def test_hash_mismatch_fails_closed(self):
        with patch.object(gradio_frontend_compat, "ORIGINAL_ASSET_SHA256", "0" * 64):
            with self.assertRaisesRegex(
                gradio_frontend_compat.GradioFrontendCompatibilityError,
                "does not match",
            ):
                gradio_frontend_compat.build_patched_tabs_asset()

    def test_tampered_prebuilt_asset_fails_closed(self):
        asset = gradio_frontend_compat.build_patched_tabs_asset()
        tampered = gradio_frontend_compat.PatchedFrontendAsset(
            filename=asset.filename,
            content=asset.content + b" ",
            original_sha256=asset.original_sha256,
            patched_sha256=asset.patched_sha256,
        )

        with self.assertRaisesRegex(
            gradio_frontend_compat.GradioFrontendCompatibilityError,
            "integrity validation",
        ):
            gradio_frontend_compat.install_gradio_tabs_compatibility_route(
                FastAPI(),
                tampered,
            )

    def test_exact_route_precedes_generic_assets_and_supports_etag(self):
        app = FastAPI()

        @app.get("/assets/{path:path}")
        async def generic_asset(path: str):
            return {"generic": path}

        asset = gradio_frontend_compat.build_patched_tabs_asset()
        installed = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app, asset)
        self.assertIs(installed, asset)
        client = TestClient(app)

        response = client.get(f"/assets/{asset.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, asset.content)
        self.assertEqual(response.headers["content-length"], str(len(asset.content)))
        self.assertEqual(response.headers["etag"], f'"{asset.patched_sha256}"')
        self.assertEqual(response.headers["x-aikimi-gradio-compat"], "6.17.3")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

        head = client.head(f"/assets/{asset.filename}")
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(head.headers["content-length"], str(len(asset.content)))
        self.assertEqual(head.headers["etag"], f'"{asset.patched_sha256}"')

        cached = client.get(
            f"/assets/{asset.filename}",
            headers={"If-None-Match": f'"{asset.patched_sha256}"'},
        )
        self.assertEqual(cached.status_code, 304)
        self.assertEqual(client.get("/assets/other.js").json(), {"generic": "other.js"})

    def test_route_is_idempotent_and_works_with_asgi_root_path(self):
        app = FastAPI(root_path="/aikimi")
        app.router.routes.insert(0, object())
        first = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app)
        second = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app)

        self.assertIs(first, second)
        self.assertEqual(
            sum(getattr(route, "name", None) == gradio_frontend_compat.PATCH_ROUTE_NAME for route in app.router.routes),
            1,
        )
        response = TestClient(app, root_path="/aikimi").get(f"/assets/{first.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, first.content)


if __name__ == "__main__":
    unittest.main()
