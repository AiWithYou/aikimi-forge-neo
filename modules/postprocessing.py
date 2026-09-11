import os
from contextlib import contextmanager
from itertools import chain

import av
import numpy as np
from PIL import Image
from tqdm import tqdm

from modules import devices, images, infotext_utils, scripts, scripts_postprocessing, shared, ui_common
from modules.extras_workflow import result_html
from modules.shared import opts


@contextmanager
def _extras_job():
    devices.torch_gc()
    shared.state.begin(job="extras")
    try:
        yield
    finally:
        try:
            shared.state.end()
        finally:
            devices.torch_gc()


@_extras_job()
def run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, _, *args, save_output: bool = True):
    outputs = []

    if isinstance(image, dict):
        image = image["composite"]

    def get_images(extras_mode, image, image_folder, input_dir):
        if extras_mode == 1:
            for img in image_folder:
                if isinstance(img, Image.Image):
                    image = img
                    fn = ""
                else:
                    image = os.path.abspath(img.name)
                    fn = os.path.splitext(img.name)[0]
                yield image, fn
        elif extras_mode == 2:
            assert not shared.cmd_opts.hide_ui_dir_config, "--hide-ui-dir-config option must be disabled"
            assert input_dir, "input directory not selected"

            image_list = shared.listfiles(input_dir)
            for filename in image_list:
                yield filename, filename
        else:
            assert image, "image not selected"
            yield image, None

    if extras_mode == 2 and output_dir != "":
        outpath = output_dir
    else:
        outpath = opts.outdir_samples or opts.outdir_extras_samples

    infotext = ""
    display_info = {}

    data_to_process = list(get_images(extras_mode, image, image_folder, input_dir))
    shared.state.job_count = len(data_to_process)

    for image_placeholder, name in data_to_process:
        image_data: Image.Image

        shared.state.nextjob()
        shared.state.textinfo = name
        shared.state.skipped = False

        if shared.state.interrupted or shared.state.stopping_generation:
            break

        if isinstance(image_placeholder, str):
            try:
                image_data = images.read(image_placeholder)
            except Exception:
                if extras_mode != 2:
                    raise
                continue
        else:
            image_data = images.fix_image(image_placeholder) if extras_mode == 1 else image_placeholder

        image_data = image_data if image_data.mode in ("RGBA", "RGB") else image_data.convert("RGB")

        parameters, existing_pnginfo = images.read_info_from_image(image_data)
        if parameters:
            existing_pnginfo["parameters"] = parameters

        initial_pp = scripts_postprocessing.PostprocessedImage(image_data)

        scripts.scripts_postproc.run(initial_pp, args)
        display_info = initial_pp.info

        if shared.state.skipped:
            continue

        used_suffixes = {}
        for pp in [initial_pp, *initial_pp.extra_images]:
            suffix = pp.get_suffix(used_suffixes)

            if opts.use_original_name_batch and name is not None:
                basename = os.path.splitext(os.path.basename(name))[0]
                forced_filename = basename + suffix
            else:
                basename = ""
                forced_filename = None

            infotext = ", ".join([k if k == v else f"{k}: {infotext_utils.quote(v)}" for k, v in pp.info.items() if v is not None])

            if opts.enable_pnginfo:
                pp.image.info = existing_pnginfo

            shared.state.assign_current_image(pp.image)

            if save_output:
                fullfn, _ = images.save_image(pp.image, path=outpath, basename=basename, extension=opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="postprocessing", existing_info=existing_pnginfo, forced_filename=forced_filename, suffix=suffix)

            if extras_mode != 2 or show_extras_results:
                outputs.append(pp.image)

    return outputs, result_html(display_info), ""


@_extras_job()
def run_postprocessing_video(_mode, _img, _folder, _in_dir, _out_dir, _show, video_input, *args, save_output: bool = True):
    from modules.video_writer import VideoEncodingCancelled

    outputs: list[np.ndarray] = []
    infotext = ""
    container = av.open(video_input)
    try:
        video_stream = container.streams.best("video")
        if video_stream is None:
            raise ValueError("The input file does not contain a video stream")
        frames = video_stream.frames
        shared.state.job_count = frames

        def processed_frames():
            nonlocal infotext
            for i, frame in enumerate(tqdm(container.decode(video_stream), desc="Processing Video", total=frames, unit="frame")):
                shared.state.nextjob()
                shared.state.textinfo = str(i)
                shared.state.skipped = False
                if shared.state.interrupted or shared.state.stopping_generation:
                    raise VideoEncodingCancelled()
                initial_pp = scripts_postprocessing.PostprocessedImage(frame.to_image())
                scripts.scripts_postproc.run(initial_pp, args)
                if shared.state.interrupted or shared.state.stopping_generation:
                    raise VideoEncodingCancelled()
                if shared.state.skipped:
                    continue
                if not outputs:
                    infotext = ", ".join([k if k == v else f"{k}: {infotext_utils.quote(v)}" for k, v in initial_pp.info.items() if v is not None])
                shared.state.assign_current_image(initial_pp.image)
                output = np.array(initial_pp.image, dtype=np.uint8)
                outputs[:] = [output]
                yield output
            if shared.state.interrupted or shared.state.stopping_generation:
                raise VideoEncodingCancelled()

        processed = processed_frames()
        try:
            first = next(processed, None)
            if first is not None:
                if save_output:
                    rate = video_stream.average_rate
                    if rate is None:
                        raise ValueError("The input video does not declare an average frame rate")
                    images.save_video(
                        os.path.splitext(os.path.basename(video_input))[0],
                        chain((first,), processed),
                        fps=rate,
                        basename=None,
                        info=infotext,
                        audio_copy=video_input,
                    )
                else:
                    for _frame in processed:
                        pass
        except VideoEncodingCancelled:
            pass
        finally:
            processed.close()
    finally:
        container.close()
    return outputs, ui_common.plaintext_to_html(infotext), ""


def run_postprocessing_webui(id_task, *args, **kwargs):
    if args[0] == 3:
        return run_postprocessing_video(*args, **kwargs)
    else:
        return run_postprocessing(*args, **kwargs)


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True, max_side_length: int = 0):
    # Handler for API (does not support video)

    args = scripts.scripts_postproc.create_args_for_run(
        {
            "Upscale": {
                "upscale_enabled": True,
                "upscale_mode": resize_mode,
                "upscale_by": upscaling_resize,
                "max_side_length": max_side_length,
                "upscale_to_width": upscaling_resize_w,
                "upscale_to_height": upscaling_resize_h,
                "upscale_crop": upscaling_crop,
                "upscaler_1_name": extras_upscaler_1,
                "upscaler_2_name": extras_upscaler_2,
                "upscaler_2_visibility": extras_upscaler_2_visibility,
            },
            "GFPGAN": {
                "enable": True,
                "gfpgan_visibility": gfpgan_visibility,
            },
            "CodeFormer": {
                "enable": True,
                "codeformer_visibility": codeformer_visibility,
                "codeformer_weight": codeformer_weight,
            },
        }
    )

    return run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, "", *args, save_output=save_output)
