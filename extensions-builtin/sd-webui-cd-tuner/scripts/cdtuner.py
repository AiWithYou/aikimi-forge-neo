import gradio as gr
import torch
import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw
import math
import random
from torch.nn import Parameter
import modules.ui
import modules
from functools import wraps
from modules import devices, shared, extra_networks
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser,CFGDenoisedParams, on_cfg_denoised

debug = False

# Aikimi adaptation of hako-mikan/sd-webui-cd-tuner, revision 3685692b6a7001d3f34c5dfbec4756ac59884047.
# Original and adapted code: AGPL-3.0; see ../LICENSE and ../UPSTREAM.json.
from modules.ui_components import InputAccordion, ToolButton
from modules_forge.cd_tuner_state import TensorEditLedger
from modules.script_callbacks import AfterCFGCallbackParams, on_cfg_after_cfg

IS_FORGE = True
denoised_params = AfterCFGCallbackParams
DEFAULTC = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0]
startup_t = startup_i = False
_ACTIVE_SCRIPT = None


def _cleanup_active():
    if _ACTIVE_SCRIPT is not None:
        _ACTIVE_SCRIPT._cleanup()


def _guarded(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            processing = getattr(self, "_processing", None)
            if processing is not None:
                processing.extra_generation_params.pop("CDT", None)
                processing.extra_generation_params.pop("CDTC", None)
            self._cleanup()
            raise
    return wrapped


def _parse_colors(text, regions):
    groups = text.replace("|", ";").split(";")
    colors = []
    for group in groups:
        values = [float(x) for x in group.replace(",", " ").split()]
        if len(values) not in (3, 4) or not all(math.isfinite(x) for x in values):
            raise ValueError("CD Tuner: 色は有限値3個（RGB）または4個（明るさ, RGB）で指定してください。")
        if len(values) == 3:
            values.insert(0, sum(values) / 3)
        colors.append(values)
    if len(colors) == 1:
        return [list(colors[0]) for _ in range(regions)]
    if len(colors) != regions:
        raise ValueError(f"CD Tuner: 色の数を領域数{regions}に合わせるか、全領域共通の1色を指定してください。")
    return colors

class Script(modules.scripts.Script):   
    def __init__(self):
        self._edits = TensorEditLedger()
        #Color/Detail
        self.active = False
        self.storedweights = {}
        self.storedweights_vae = {}
        self.storedweights_vae2 = {}
        self.shape = None
        self.done = [False,False]
        self.pas = 0
        self.colored = 0
        self.randman = None
        self.saturation = 0
        self.saturation2 = 0

        self.adjusts = list(ADJUSTS)
        self.bias_layout = None
        self.latent_channels = 4
        self.bright_dir = None
        self.vaekeys = list(VAEKEYS)
        self.vaekeys2 = list(VAEKEYS2)

        #Color map
        self.activec = False
        self.ocells = []
        self.icells = []
        self.cmode = ""
        self.colors =[]

    def title(self):
        return "CD Tuner"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    infotext_fields = None
    paste_field_names = []

    def ui(self, is_img2img):      
        resetsymbol = '\U0001F5D1\U0000FE0F'

        with InputAccordion(startup_i if is_img2img else startup_t, label=self.title()) as active:
            gr.Markdown("生成中の色・質感を調整します。既定はオフ。まず小さい値で同じSeedを比較してください。設定は通常のUI設定保存を使用します。")
            with gr.Tab("Color/Detail"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            d1 = gr.Slider(label="Detail(d1)", minimum=-10, maximum=10, value=0.0, step=0.1)
                            refresh_d1 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            d2 = gr.Slider(label="Detail 2(d2)", minimum=-10, maximum=10, value=0.0, step=0.1,visible = True)
                            refresh_d2 = ToolButton(value=resetsymbol)
                    with gr.Column():
                        with gr.Row():
                            hd1 = gr.Slider(label="hr-Detail(hd1)", minimum=-10, maximum=10, value=0.0, step=0.1)
                            refresh_hd1 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            hd2 = gr.Slider(label="hr-Detail 2(hd2)", minimum=-10, maximum=10, value=0.0, step=0.1,visible = True)
                            refresh_hd2 = ToolButton(value=resetsymbol)
                with gr.Row():
                    pass
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            cont1 = gr.Slider(label="Contrast(con1)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_cont1 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            cont2 = gr.Slider(label="Contrast 2(con2)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_cont2 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            bri = gr.Slider(label="Brightness(bri)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_bri = ToolButton(value=resetsymbol)
                        with gr.Row():
                            sat = gr.Slider(label="saturation(sat)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_sat = ToolButton(value=resetsymbol)
                    with gr.Column():
                        with gr.Row():
                            col1 = gr.Slider(label="Cyan-Red(col1)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_col1 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            col2 = gr.Slider(label="Magenta-Green(col2)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_col2 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            col3 = gr.Slider(label="Yellow-Blue(col3)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_col3 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            sat2 = gr.Slider(label="saturation2(sat2)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_sat2 = ToolButton(value=resetsymbol)

                    scaling = gr.Checkbox(value=False, label="hr-scaling(hrs)",interactive=True,elem_id=f"cdt-{is_img2img}-hr-scaling")
                    stop = gr.Slider(label="Stop Step", minimum=-1, maximum=20, value=-1, step=1)
                    stoph = gr.Slider(label="Hr-Stop Step", minimum=-1, maximum=20, value=-1, step=1)
                with gr.Row():
                    opts = gr.CheckboxGroup(choices=["Apply once"],show_label = False)

            with gr.Tab("Color Map"):
                with gr.Row():
                    cmode = gr.Radio(label="Split by", choices=["Horizontal","Vertical"], value="Horizontal", type="value", interactive=True)
                    ratios = gr.Textbox(label="Split Ratio",lines=1,value="1,1",interactive=True,elem_id=f"cdt-{is_img2img}-split-ratio",visible=True)
                with gr.Row():
                    with gr.Column():
                        maketemp = gr.Button(value="Visualize Map")
                        colors = gr.Textbox(label="colors",interactive=True,visible=True)
                        fst = gr.Slider(label="Mapping Stop Step", minimum=0, maximum=150, value=2, step=1)
                        att = gr.Slider(label="Strength", minimum=0, maximum=2, value=1, step=0.1)
                        presets = gr.Dropdown(label="add from presets", choices=list(COLORPRESET.keys()))
                    
                    presets.change(fn=lambda x, y:COLORPRESET[y] if x == "" else x + ";"+ COLORPRESET[y], inputs =[colors,presets], outputs=[colors])
                    
                    with gr.Column():
                        areasimg = gr.Image(type="pil", show_label = False, height = 256, width = 256)
                    
                    dtrue =  gr.Checkbox(value = True, visible = False)                
                    dfalse =  gr.Checkbox(value = False,visible = False)     

                maketemp.click(fn=makecells, inputs =[ratios,cmode,dtrue],outputs = [areasimg])

            def infotexter(text):
                if text == "":return [gr.update()] * (len(params) +1)
                if debug : print(text)
                text = text.split(",")
                if len(text) == 9:
                    outs = [float(x) for x in text[0:3]] +[0] * 2 + [float(x) for x in text[3:8]] + [text[-1] == "1"] + [-1] *2 + [0,0]
                    outs.insert(3,0)
                elif len(text) == 13:
                    outs = [float(x) for x in text[:10]] + [text[10] == "1"] + [float(x) for x in text[11:]] + [0,0]
                elif len(text) in (14, 15, 16):
                    # 保存形式の14番目は廃止済みのdis。UIには対応する入力がない。
                    saturation = [float(x) for x in text[14:16]]
                    outs = (
                        [float(x) for x in text[:10]]
                        + [float(text[10]) == 1]
                        + [float(x) for x in text[11:13]]
                        + saturation
                        + [0] * (2 - len(saturation))
                    )
                else:
                    outs = [0] * 10 + [False] + [-1] *2 + [0, 0]
                outs = [""] + [True] + outs
                return [gr.update(value = x) for x in outs]

            def infotexter_c(text):
                if text == "":return [gr.update()] * (len(paramsc) +1)
                if debug : print(text)
                text = text.split("_")
                if len(text) != 5:
                    text = ["","Horizonal","",2,0.2]
                if debug : print(text)
                text = [""] + text
                return [gr.update(value = x) for x in text]

            refresh_d1.click(fn=lambda:gr.update(value = 0),outputs=[d1], show_progress=False)
            refresh_d2.click(fn=lambda:gr.update(value = 0),outputs=[d2], show_progress=False)
            refresh_hd1.click(fn=lambda:gr.update(value = 0),outputs=[hd1], show_progress=False)
            refresh_hd2.click(fn=lambda:gr.update(value = 0),outputs=[hd2], show_progress=False)
            refresh_cont1.click(fn=lambda:gr.update(value = 0),outputs=[cont1], show_progress=False)
            refresh_cont2.click(fn=lambda:gr.update(value = 0),outputs=[cont2], show_progress=False)
            refresh_col1.click(fn=lambda:gr.update(value = 0),outputs=[col1], show_progress=False)
            refresh_col2.click(fn=lambda:gr.update(value = 0),outputs=[col2], show_progress=False)
            refresh_col3.click(fn=lambda:gr.update(value = 0),outputs=[col3], show_progress=False)
            refresh_bri.click(fn=lambda:gr.update(value = 0),outputs=[bri], show_progress=False)
            refresh_sat.click(fn=lambda:gr.update(value = 0),outputs=[sat], show_progress=False)
            refresh_sat2.click(fn=lambda:gr.update(value = 0),outputs=[sat2], show_progress=False)

            params = [active,d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,scaling,stop,stoph,sat,sat2]
            paramsc = [ratios,cmode,colors,fst,att]

            allsets = gr.Textbox(visible = False, value="")
            allsets.change(fn=infotexter,inputs = [allsets], outputs =[allsets] + params)

            allsets_c = gr.Textbox(visible = False, value="")
            allsets_c.change(fn=infotexter_c,inputs = [allsets_c], outputs =[allsets_c] +paramsc)

        self.infotext_fields = ([(allsets,"CDT"),(allsets_c,"CDTC")])
        self.paste_field_names = ["CDT", "CDTC"]

        return params + paramsc + [opts]

    @_guarded
    def process_batch(self, p, active, d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,scaling,stop,stoph,sat,sat2,ratios,cmode,colors,fst,att,opts,**kwargs):
        _cleanup_active()
        p.extra_generation_params.pop("CDT", None)
        p.extra_generation_params.pop("CDTC", None)
        if "Apply once" in opts and getattr(self, "_once_request", None) is p:
            return
        if (self.done[0] or self.done[1]) and self.storedweights and self.storedname == shared.opts.sd_model_checkpoint:
            restoremodel(self)

                # ["d1","d2","con1","con2","bri","col1","col2","col3","hd1","hd2","hrs","st1","st2","dis","sat"]
                     #0  1   2       3        4   5     6     7     8     9    10                      11    12    13 14
        allsets = [d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,1 if scaling else 0,stop,stoph,0,sat,sat2]
        allsets_c = [ratios,cmode,colors,fst,att]
        self.opts = opts

        self.__init__()

        if not active:
            return

        global _ACTIVE_SCRIPT
        _ACTIVE_SCRIPT = self
        self._processing = p
        if "Apply once" in opts:
            self._once_request = p
        prompts = kwargs.get("prompts") or getattr(p, "prompts", None) or p.all_prompts
        psets, psets_c = fromprompts(prompts[:1])

        for i, param in enumerate(psets):
            if param is not None:
                allsets[i] = param
        
        for i, param in enumerate(psets_c):
            if param is not None:
                if param == "H" :param ="Horizontal"
                if param == "V" :param ="Vertical"
                allsets_c[i] = param

        ratios, cmode, colors, fst, att = allsets_c

        if debug: print("\n",allsets)
        if debug: print("\n",allsets_c)

        # "conditioner" only exists on the A1111 backend, on Forge an SDXL model
        # was detected as SD1.5 and used the wrong colour table
        self.isxl = getattr(shared.sd_model, "is_sdxl", False) or hasattr(shared.sd_model,"conditioner")
        self.adjusts, self.bias_layout = get_adjusts(shared.sd_model)
        self.latent_channels = latent_channels(shared.sd_model)
        self.bright_dir = brightness_direction(shared.sd_model)
        self.vaekeys, self.vaekeys2 = vae_keys(shared.sd_model)

        self.isrefiner = getattr(p, "refiner_switch_at", None) is not None

        if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in allsets):
            raise ValueError("CD Tuner: 調整値は有限の数値で指定してください。")
        if not math.isfinite(float(att)) or not 0 <= float(att) <= 2 or not math.isfinite(float(fst)) or float(fst) != int(fst) or not 0 <= int(fst) <= 150:
            raise ValueError("CD Tuner: Color Map強度は0〜2、停止Stepは0〜150の整数で指定してください。")
        if allsets == DEFAULTC and (att == 0 or colors == ""):return
        else:
            if not all(x == 0 for x in allsets[:10] + allsets[14:16]):
                if debug:print("Start")
                self.active = True
                self.drratios = allsets[0:3]+allsets[8:10]
                self.ddratios= allsets[3:8]
                self.scaling = allsets[10]
                self.sts = allsets[11:14]
                self.saturation = allsets[14]
                self.saturation2 = allsets[15]
                if hasattr(p,"enable_hr"): # Img2img doesn't have it.
                    self.hr = p.enable_hr
                else:
                    self.hr = False

                p.extra_generation_params.update({"CDT":",".join([str(x) for x in (allsets[:-2] + allsets[-2:])])})

                self.storedname = shared.opts.sd_model_checkpoint

            if colors.strip() and float(att) != 0:
                self.activec = True
                if not ("Hor" in cmode or "Ver" in cmode) :cmode = "Vertical"
                self.cmode = cmode
                self.fst = int(fst)
                self.att = float(att) * 0.5
                self.batch = p.batch_size
                self.ocells, self.icells = makecells(ratios, cmode,False)
                self.colors = _parse_colors(colors, sum(len(cells) for cells in self.icells))

                if 5 > self.fst :
                    self.satt = 1.15 - self.fst * 0.15
                else:
                    self.satt = 0.4

                p.extra_generation_params.update({"CDTC":"_".join([str(x) for x in allsets_c])})

            if self.saturation != 0:
                vaedealer(self)

            if self.saturation2 != 0:
                vaedealer2(self)

        print(f"\nCD Tuner Effective : {allsets}")

        if not hasattr(self,"cdt_dr_callbacks"):
            self.cdt_dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

        if IS_FORGE:
            if not hasattr(self,"cdt_dd_callbacks"):            
                self.cdt_dd_callbacks = on_cfg_after_cfg(self.denoised_callback)
        else:
            if not hasattr(self,"cdt_dd_callbacks"):
                self.cdt_dd_callbacks = on_cfg_denoised(self.denoised_callback)

    def postprocess_batch(self, p, *args, **kwargs):
        self._cleanup()


    def _cleanup(self):
        global _ACTIVE_SCRIPT
        self.active = self.activec = False
        self._edits.restore()
        self.storedweights.clear()
        self.storedweights_vae.clear()
        self.storedweights_vae2.clear()
        self.done = [False, False]
        self._processing = None
        if _ACTIVE_SCRIPT is self:
            _ACTIVE_SCRIPT = None


    def on_process_cleanup(self, p, *args):
        try:
            self._cleanup()
        finally:
            self._once_request = None

    @_guarded
    def denoiser_callback(self, params: CFGDenoiserParams):
        # params.x [batch,ch[4],height,width]
        #print(self.activec,self.colors,self.ocells,self.icells,params.sampling_step)
        if self.activec:
            if self.shape is None:self.shape = params.x.shape
            if getattr(self._processing, "is_hr_pass", False) and not self.pas:
                self.colored = 0
                self.pas = 1
            # Forgeのコールバックはcond/uncond分岐前に1回呼ばれる。
            # Hires移行時だけカウントを戻し、各Stepで一度ずつ適用する。
            cond_last = params.x.shape[0] > self.batch
            passes = 1 if IS_FORGE or cond_last else 2
            if self.colored // passes == params.sampling_step and self.colored // passes < self.fst:
                c = 0
                scale = torch.mean(torch.abs(params.x[:,:,:,:]))
                h,w = params.x.shape[-2], params.x.shape[-1]
                # the Wan latent of Anima and Krea2 has a frame dimension, so the
                # region is always addressed through the last two dimensions
                enhance = 6
                hr_att = 0.25 if self.pas else 1
                for i,ocell in enumerate(self.ocells):
                    for icell in self.icells[i]:
                        if "Ver" in self.cmode:
                            s3 = slice(int(h*ocell[0]), int(h*ocell[1]))
                            s4 = slice(int(w*icell[0]), int(w*icell[1]))
                        elif "Hor" in self.cmode:
                            s3 = slice(int(h*icell[0]),int(h*icell[1]))
                            s4 = slice(int(w*ocell[0]),int(w*ocell[1]))

                        colorvec = colorcalc(self.colors[c],self.isxl,shared.sd_model)
                        if len(colorvec) > 3:
                            peak = max(abs(x) for x in colorvec) or 1.0
                            wanted = MAP_PEAK * sum(abs(x) for x in self.colors[c])
                            colorvec = [x / peak * wanted for x in colorvec]
                            # 16 channel latent: every channel carries colour, and the
                            # magnitude is normalised so a slider value pushes as hard
                            # as it does on a 4 channel one
                            targets = list(range(min(len(colorvec), params.x.shape[1])))
                            offsets = list(colorvec)
                            weight = 3.0 / len(colorvec)
                            # the 4 channel path leaves the luminance channel alone and
                            # only blends the colour ones. Every channel of a 16 channel
                            # latent carries structure, so blending them all replaces the
                            # image with a flat colour: add the shift instead
                            keep = 1.0
                        else:
                            targets = [1,2,3]
                            offsets = [0.0] + list(colorvec)
                            weight = 1.0
                            keep = None
                        cratio =(sum(abs(x * 50) for x in colorvec))/10 * weight * (1/(1+(1 + params.sampling_step)**1.5/10)) * self.att * hr_att * self.satt
                        if 0 > cratio :
                            c += 1
                            continue
                        for s2 in targets:
                            scale = torch.mean(torch.abs(params.x[:,s2]))
                            blend = (1 - cratio) if keep is None else keep
                            shift = offsets[s2]*enhance * scale * cratio
                            params.x[:-self.batch,s2,...,s3,s4] = blend * params.x[:-self.batch,s2,...,s3,s4] - shift
                            params.x[-self.batch:,s2,...,s3,s4] = blend * params.x[-self.batch:,s2,...,s3,s4] + (shift if cond_last else -shift)

                        c += 1
                self.colored += 1

        if self.active and any(self.drratios):
            if debug: print(self.drratios)
            if self.shape is None:self.shape = params.x.shape
            if params.x.shape[-2] * params.x.shape[-1] > self.shape[-2]*self.shape[-1]:
                self.pas = 1
                scale = ((params.x.shape[-2] * params.x.shape[-1]) / (self.shape[-2]*self.shape[-1])) if self.scaling else 1
            else: 
                scale = 1

            ratios = list(self.drratios)

            if stopper(self,self.pas,params.sampling_step): return

            #dont operate twice
            if self.done[self.pas] and not self.isrefiner:
                return
            else:
                self.done[self.pas] = True

            if self.pas:
                if ratios[-2] : ratios[0] = ratios[-2]
                if ratios[-1] : ratios[1] = ratios[-1]

            ratios[:2] = [x * scale for x in ratios[:2]]
            ratios = fineman(ratios)
            if debug: print(ratios)
            for i,name in enumerate(self.adjusts):
                if name is None or (i < 4 and ratios[i] == 1) or (i == 4 and ratios[i][0] == 0): continue
                if name not in self.storedweights.keys() or self.isrefiner:
                    self.storedweights[name] = getset_nested_module_tensor(True, shared.sd_model, name).clone()
                # Work in the parameter's own dtype. A DiT usually runs in bfloat16
                # while devices.dtype is float16, and mixing the two raises in the
                # matmul. float8 keeps the old behaviour of computing in and writing
                # back devices.dtype.
                dtype = self.storedweights[name].dtype
                if dtype == torch.float8_e4m3fn:
                    work = dtype = devices.dtype
                else:
                    work = dtype
                base = self.storedweights[name].to(devices.device, work)
                if 4 > i:
                    new_weight = base * torch.tensor(ratios[i]).to(devices.device, work)
                else:
                    offset = ratios[i]
                    if self.bias_layout is not None:
                        offset = bias_offset(offset[0], self.storedweights[name].shape[0], self.latent_channels, self.bias_layout, self.bright_dir)
                    new_weight = base + torch.as_tensor(offset).to(devices.device, work)
                getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = new_weight.to(dtype))
            
            self.shape = params.x.shape


    @_guarded
    def denoised_callback(self, params: denoised_params):
        if self.active:
            if self.isrefiner:
                restoremodel(self)
            if self.hr and not getattr(self._processing, "is_hr_pass", False):
                return
            if params.sampling_step == params.total_sampling_steps-2 -self.sts[2]: 
                if any(x != 0 for x in self.ddratios):
                    colors = colorcalc(self.ddratios[1:],self.isxl,shared.sd_model)
                    if len(colors) > 3:
                        # 16 channel latent, the colour shift covers every channel and
                        # cont2 keeps acting on the first one
                        ratios = list(colors)
                        ratios[0] += self.ddratios[0] * 0.02
                    else:
                        ratios = [self.ddratios[0] * 0.02] + colors
                    print(f"\nCD Tuner After Generation Effective: {ratios}")
                    for i, x in enumerate(ratios):
                        if i >= params.x.shape[1]: break
                        params.x[:,i,:,:] = params.x[:,i,:,:] - x * 20/3

def vae_decoder(sd_model):
    try:
        vae = sd_model.forge_objects_after_applying_lora.vae if IS_FORGE else sd_model
        return vae.first_stage_model.decoder
    except AttributeError:
        return None

def vae_keys(sd_model):
    """The decoder weights the two saturation sliders scale.

    The SD and the Flux autoencoders share the same decoder layout, the Wan
    autoencoder used by Anima and Krea2 does not: its decoder is a flat Sequential
    of residual blocks and resamplers, so the equivalent layers are looked up by
    type instead of by a fixed path.
    """
    decoder = vae_decoder(sd_model)
    if decoder is None or type(decoder).__name__ != "Decoder3d":
        return list(VAEKEYS), list(VAEKEYS2)

    root = f"{forge_prefix_v}first_stage_model.decoder."
    blocks = list(decoder.upsamples)
    resamples = [i for i, m in enumerate(blocks) if type(m).__name__ == "Resample"]
    shortcuts = [i for i, m in enumerate(blocks)
                 if type(m).__name__ == "ResidualBlock" and hasattr(getattr(m, "shortcut", None), "weight")]

    keys, keys2 = [], []
    if resamples:
        # the resampler closest to the output, like up.1.upsample on the SD decoder
        keys.append(f"{root}upsamples.{resamples[-1]}.resample.1.weight")
        keys2 += [f"{root}upsamples.{i}.resample.1.weight" for i in resamples[:-1]]
    late = [i for i in shortcuts if not resamples or i > resamples[-1]]
    if late:
        keys.append(f"{root}upsamples.{late[0]}.shortcut.weight")
    keys2 += [f"{root}conv1.weight", f"{root}head.2.weight"]
    return keys, keys2

def vaedealer(self):
    for name in self.vaekeys:
        if name not in self.storedweights_vae:
            self.storedweights_vae[name] = getset_nested_module_tensor(True, shared.sd_model, name).clone()
        new_weight = self.storedweights_vae[name].to(devices.device) * (1 + self.saturation * 0.075) 
        getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = new_weight)

def vaeunloader(self):
    for name in self.storedweights_vae:
        getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = self.storedweights_vae[name])

def vaedealer2(self):
    for name in self.vaekeys2:
        if name not in self.storedweights_vae2:
            self.storedweights_vae2[name] = getset_nested_module_tensor(True, shared.sd_model, name).clone()
        new_weight = self.storedweights_vae2[name].to(devices.device) * (1 + self.saturation2 * 0.02) 
        getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = new_weight)

def vaeunloader2(self):
    for name in self.storedweights_vae2:
        getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = self.storedweights_vae2[name])

def stopper(self,pas,step):
    judge = False
    if 0 > self.sts[pas]: return False 
    if step >= self.sts[pas]:
        judge = True
    if judge and self.done[pas]:
        restoremodel(self)
        self.done[pas] = False
    return judge


def getset_nested_module_tensor(clone,model, tensor_path, new_tensor = None):
    sdmodules = tensor_path.split('.')
    target_module = model
    last_attr = None

    for module_name in sdmodules if clone else sdmodules[:-1]:
        if module_name.isdigit():
            target_module = target_module[int(module_name)] 
        else:
            target_module = getattr(target_module, module_name) 

    if clone : return target_module

    last_attr = sdmodules[-1]
    current = getattr(target_module, last_attr, None)
    if not isinstance(current, torch.Tensor) or not current.is_floating_point():
        raise ValueError("CD Tuner: この量子化層の重み編集には未対応です。Detail/Contrast 1/Saturationを0に戻してください。")
    if _ACTIVE_SCRIPT is not None:
        _ACTIVE_SCRIPT._edits.capture(target_module, last_attr, group="vae" if ".vae." in tensor_path else "diffusion")

    if isinstance(current, torch.Tensor) and current.shape == new_tensor.shape:
        # Write into the existing tensor instead of replacing it. A fresh Parameter
        # is not the object the Forge memory manager captured when it loaded the
        # model, so it is skipped when the model is streamed to the CPU and stays
        # behind on the GPU. The manager then believes it freed memory it did not,
        # and the next generation fails inside the module move and is reported as an
        # OOM even with the GPU nearly empty. Keeping the tensor also keeps its
        # device, which matters while the model is offloaded.
        # the backend loads the model under torch.inference_mode(), so its weights are
        # inference tensors and can only be written to from inside that mode
        with torch.inference_mode():
            current.copy_(new_tensor.to(device=current.device, dtype=current.dtype))
        return

    setattr(target_module, last_attr, Parameter(new_tensor, requires_grad=False)) 

def restoremodel(self):
    self._edits.restore(group="diffusion")
    self.storedweights.clear()

def fineman(fine):
    fine = [
        1 - fine[0] * 0.01,
        1+ fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1+ fine[1] * 0.02,
        [fine[2] * 0.02,0,0,0]
                ]
    return fine

def colorcalc(cols,isxl,sd_model = None):
    # cols is [brightness, red, green, blue]
    pinv = latent_rgb_pinv(sd_model) if sd_model is not None else None
    if pinv is None:
        colors = COLSXL if isxl else COLS
        outs = [[y * cols[i] * 0.02 for y in x] for i,x in enumerate(colors)]
        return [sum(x) for x in zip(*outs)]
    drgb = np.array([cols[0] + cols[1], cols[0] + cols[2], cols[0] + cols[3]], dtype=np.float64)
    # the caller subtracts the result, the sign is baked into COLS/COLSXL the same way
    return [-float(x) * 0.02 for x in pinv @ drgb]

def fromprompts(prompt):
    _, extra_network_data = extra_networks.parse_prompts(prompt)
    params = extra_network_data["cdt"] if "cdt" in extra_network_data.keys() else None
    params_c = extra_network_data["cdtc"] if "cdtc" in extra_network_data.keys() else None

    outs = [None] * len(IDENTIFIER) 
    outs_c = [None] * len(IDENTIFIER_C) 

    if params is None and params_c is None : return outs, outs_c
    
    if params:
        params = params[0].items[0]
        if any(x in params for x in IDENTIFIER):
            params = params.split(";")
            for p in params:
                if "=" in p:
                    p = [x.strip() for x in p.split("=")]
                    if p[0] in IDENTIFIER:
                        if p[0] == "dis": return [0] * len(IDENTIFIER), outs_c
                        outs[IDENTIFIER.index(p[0])] = float(p[1])
        else:
            params = [float(x) for x in params.split(";")]
            outs[:len(params)] = params

    if params_c:
        params_c = params_c[0].items[0]
        if any(x in params_c for x in IDENTIFIER_C):
            params_c = params_c.split(";")
            for p in params_c:
                if "=" in p:
                    p = [x.strip() for x in p.split("=")]
                    if p[0] in IDENTIFIER_C:
                        outs_c[IDENTIFIER_C.index(p[0])] = p[1]
        else:
            params_c = [x for x in params_c.split(";")]
            outs_c[:len(params_c)] = params_c

    if outs[10] is not None and outs[10] != 0: outs[10] = 1
    if debug : print(outs)
    if debug : print(outs_c)
    return outs, outs_c

def lange(l):
    return range(len(l))

def makecells(aratio,mode,in_ui):
    if not isinstance(aratio, str) or not aratio.strip():
        raise ValueError("CD Tuner: 分割比を入力してください。")
    if not ("Hor" in mode or "Ver" in mode) :mode = "Vertical"
    aratio = aratio.replace(",", " ")
    aratios = aratio.split("|") if "|" in aratio else aratio.split(";")
    if len(aratios) == 1 : aratios[0] = "1 " + aratios[0]
    h = w = 128
    icells = []
    ocells = []

    def startend(lst):
        o = []
        s = 0
        if not lst or any(not math.isfinite(x) or x <= 0 for x in lst):
            raise ValueError("CD Tuner: 分割比は正の有限値で指定してください。")
        total = sum(lst)
        if not math.isfinite(total):
            raise ValueError("CD Tuner: 分割比の合計が大きすぎます。")
        lst = [value / total for value in lst]
        for i in lange(lst):
            if i == 0 :o.append([0,lst[0]])
            else : o.append([s, s + lst[i]])
            s = s + lst[i]
        return o

    for rc in aratios:
        rc = rc.split()
        rc = [float(r) for r in rc]
        if len(rc) == 1 : rc = [rc[0]]*2
        ocells.append(rc[0])
        icells.append(startend(rc[1:]))

    if sum(map(len, icells)) > 64:
        raise ValueError("CD Tuner: Color Mapは64領域以内にしてください。")
    ocells = startend(ocells)
    if not in_ui:
        return ocells, icells
    fx = np.zeros((h,w, 3), np.uint8)
    palette = random.Random(0)

    c = 0
    for i,ocell in enumerate(ocells):
        for icell in icells[i]:
            if "Vertical" in mode:
                fx[int(h*ocell[0]):int(h*ocell[1]),int(w*icell[0]):int(w*icell[1]),:] = [palette.randint(0,255),palette.randint(0,255),palette.randint(0,255)]
            elif "Horizontal" in mode: 
                fx[int(h*icell[0]):int(h*icell[1]),int(w*ocell[0]):int(w*ocell[1]),:] = [palette.randint(0,255),palette.randint(0,255),palette.randint(0,255)]
            c += 1
    img = PIL.Image.fromarray(fx)
    draw = PIL.ImageDraw.Draw(img)
    c = 0
    def wbdealer(col):
        if sum(col) > 380:return "black"
        else:return "white"

    for i,ocell in enumerate(ocells):
        for j,icell in enumerate(icells[i]):
            if "Vertical" in mode:
                draw.text((int(w*icell[0]),int(h*ocell[0])),f"{c}",wbdealer(fx[int(h*ocell[0]),int(w*icell[0])]))
            elif "Horizontal" in mode: 
                draw.text((int(w*ocell[0]),int(h*icell[0])),f"{c}",wbdealer(fx[int(h*icell[0]),int(w*ocell[0])]))
            c += 1
    if in_ui:
        return img
    else:
        print(ocells,icells)
        return ocells, icells


def latentfromrgb(rgb):
    outs = [[y * rgb[i] for y in x] for i,x in enumerate(COLS[1:])]
    return [sum(x) for x in zip(*outs)]

forge_prefix = "forge_objects_after_applying_lora.unet." if IS_FORGE else ""
forge_prefix_v = "forge_objects_after_applying_lora.vae." if IS_FORGE else ""

ADJUSTS =[
f"{forge_prefix}model.diffusion_model.input_blocks.0.0.weight",
f"{forge_prefix}model.diffusion_model.input_blocks.0.0.bias",
f"{forge_prefix}model.diffusion_model.out.0.weight",
f"{forge_prefix}model.diffusion_model.out.0.bias",
f"{forge_prefix}model.diffusion_model.out.2.bias",
]

NAMES =[
"model.diffusion_model.input_blocks.0.0",
"model.diffusion_model.out.0",
"model.diffusion_model.out.2",
]

IDENTIFIER = ["d1","d2","con1","con2","bri","col1","col2","col3","hd1","hd2","hrs","st1","st2","dis","sat","sat2"]
IDENTIFIER_C = ["sp","md","cols","stc","str"]


COLS = [[-1,1/3,2/3],[1,1,0],[0,-1,-1],[1,0,1]]
COLSXL = [[0,0,1],[1,0,0],[-1,-1,0],[-1,1,0]]

# Peak of the colour map offset per unit of slider, taken from what COLS produces.
# The colour map amplifies by enhance * cratio and cratio is itself derived from the
# length of the offset vector, so the effect grows with the square of that length.
# A 16 channel latent needs much larger latent steps for the same RGB change, which
# would overshoot by an order of magnitude, so the vector is rescaled to this peak.
MAP_PEAK = 0.0113

# ---- Forge Neo : DiT architectures -------------------------------------------
# The UNet reads the latent with a conv, ends with a group norm and writes the
# latent back with a second conv. A DiT only has two projections, so d2 acts on
# the output projection instead of the final norm and the brightness offset goes
# into that same bias.
# The bias covers patch*patch*channels values. "block" means it is laid out as
# [channel][patch*patch], "interleave" as [patch*patch][channel].
#   diffusion model class            : (input, output, bias layout)
DIT_LAYERS = {
    "IntegratedFluxTransformer2DModel": ("img_in", "final_layer.linear", "block"),
    "IntegratedChromaTransformer2DModel": ("img_in", "final_layer.linear", "block"),
    "NextDiT": ("x_embedder", "final_layer.linear", "interleave"),
    "Anima": ("x_embedder.proj.1", "final_layer.linear", "interleave"),
    "SingleStreamDiT": ("first", "last.linear", "block"),
}

def diffusion_model(sd_model):
    try:
        if IS_FORGE:
            return sd_model.forge_objects_after_applying_lora.unet.model.diffusion_model
        return sd_model.model.diffusion_model
    except AttributeError:
        return None

def has_param(module, path):
    for name in path.split("."):
        if name.isdigit():
            try:
                module = module[int(name)]
            except (IndexError, TypeError, KeyError):
                return False
        else:
            module = getattr(module, name, None)
        if module is None:
            return False
    return True

def get_adjusts(sd_model):
    """The weight paths to touch, in the ADJUSTS order, plus the bias layout.

    Entries the model does not have are None and get skipped. On a DiT the
    output bias is a single tensor, so the d2 multiplication on it is dropped and
    only the brightness offset is applied.
    """
    dm = diffusion_model(sd_model)
    name = type(dm).__name__ if dm is not None else ""
    if name not in DIT_LAYERS:
        return list(ADJUSTS), None
    src, dst, layout = DIT_LAYERS[name]
    root = f"{forge_prefix}model.diffusion_model."
    names = [src + ".weight", src + ".bias", dst + ".weight", None, dst + ".bias"]
    return [(root + n if n and has_param(dm, n) else None) for n in names], layout

def latent_channels(sd_model):
    fmt = getattr(getattr(sd_model, "model_config", None), "latent_format", None)
    return getattr(fmt, "latent_channels", 4) if fmt is not None else 4

def bias_offset(value, length, channels, layout, direction = None):
    """Offset for the output projection bias.

    On the UNet this moves latent channel 0, which is the luminance channel of the
    4 channel VAEs. A 16 channel latent has no such channel, so ``direction`` holds
    a per channel weight and the shift follows the brightness direction instead.
    """
    offset = torch.zeros(length)
    if not channels or length % channels:
        offset[0] = value
        return offset
    if direction is None:
        direction = [1.0] + [0.0] * (channels - 1)
    per = length // channels
    for c in range(min(channels, len(direction))):
        weight = value * float(direction[c])
        if weight == 0: continue
        if layout == "interleave":
            offset[c::channels] = weight
        else:
            offset[c * per:(c + 1) * per] = weight
    return offset

def brightness_direction(sd_model):
    """Per channel weights of an equal RGB shift, scaled so the largest is 1."""
    pinv = latent_rgb_pinv(sd_model)
    if pinv is None:
        return None
    vector = pinv @ np.array([1.0, 1.0, 1.0])
    peak = np.abs(vector).max()
    return list(vector / peak) if peak else None

_RGB_PINV = {}

def latent_rgb_pinv(sd_model):
    """Map an RGB shift onto a latent shift.

    ``latent_rgb_factors`` is the least squares fit of latent -> RGB that ComfyUI
    uses for the fast preview, and it is the published mapping for the 16 channel
    VAEs. rgb = z @ M, so the smallest latent shift for a wanted RGB shift d is
    z = M @ inv(M.T @ M) @ d.
    """
    fmt = getattr(getattr(sd_model, "model_config", None), "latent_format", None)
    factors = getattr(fmt, "latent_rgb_factors", None)
    if not factors or len(factors) <= 4:
        return None
    key = type(fmt).__name__
    if key not in _RGB_PINV:
        M = np.array(factors, dtype=np.float64)
        _RGB_PINV[key] = M @ np.linalg.inv(M.T @ M)
    return _RGB_PINV[key]

def restoremodel_l(model):
    for name, module in model.named_modules():
        if name not in NAMES: continue
        if hasattr(module, "network_weights_backup"):
            if module.network_weights_backup is not None:
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.in_proj_weight.copy_(module.network_weights_backup[0])
                    module.out_proj.weight.copy_(module.network_weights_backup[1])
                else:
                    module.weight.copy_(module.network_weights_backup)
                del module.network_weights_backup
                if hasattr(module, "network_current_names"): del module.network_current_names

        if hasattr(module, "network_bias_backup"):
            if module.network_bias_backup is not None:
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.in_proj_bias.copy_(module.network_bias_backup[0])
                    module.out_proj.bias.copy_(module.network_bias_backup[1])
                else:
                    module.bias.copy_(module.network_bias_backup)
                del module.network_bias_backup
                if hasattr(module, "network_current_names"): del module.network_current_names


COLORPRESET = {
    "Cyan": "-5, 0, 0",
    "Magenda": "0, -5, 0",
    "Yellow": "0, 0, -5",
    "Red": "5, -5, -5",
    "Green": "0, 5, 0",
    "Blue": "0, 0, 5",
    "VeryLightBlue": "-5, -5, 0",
    "YellowGreen": "-5, 0, -5",
    "Orange": "0, -5, -5",
    "Malachite": "-5, 5, 0",
    "BrightCyan": "-5, 0, 5",
    "VioretPink": "0, -5, 5",
    "DeepPink": "5, -5, 0",
    "GuardsmanRed": "5, 0, -5",
    "DeepGreen": "0, 5, -5",
    "Black": "5, 5, 0",
    "BrightNeonPink": "5, 0, 5",
    "NeonBlue": "0, 5, 5",
    "GreenYellow": "-5, -5, -5",
    "LightBlue": "-5, -5, 5",
    "LimeGreen": "-5, 5, -5",
    "DeepBrown": "5, 5, -5",
    "VioletBlue": "5, 5, 5"
}

VAEKEYS = [
f"{forge_prefix_v}first_stage_model.decoder.up.1.upsample.conv.weight",
f"{forge_prefix_v}first_stage_model.decoder.up.0.block.0.nin_shortcut.weight",
]

VAEKEYS2 = [
f"{forge_prefix_v}first_stage_model.decoder.up.3.upsample.conv.weight",
f"{forge_prefix_v}first_stage_model.decoder.up.2.upsample.conv.weight",
f"{forge_prefix_v}first_stage_model.decoder.up.1.block.0.nin_shortcut.weight",
f"{forge_prefix_v}first_stage_model.decoder.conv_in.weight",
f"{forge_prefix_v}first_stage_model.decoder.conv_out.weight",
]



from modules.script_callbacks import on_script_unloaded
on_script_unloaded(_cleanup_active)
