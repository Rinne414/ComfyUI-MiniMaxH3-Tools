"""Save / load CONDITIONING to output/conditioning/<name>.safetensors (tensors + JSON metadata, no pickle)."""

import os

import folder_paths
from comfy_api.latest import io, ui

if __package__:
    from .conditioning_io import (
        clean_name,
        free_path,
        load_conditioning,
        save_conditioning,
    )
else:
    from conditioning_io import (
        clean_name,
        free_path,
        load_conditioning,
        save_conditioning,
    )

SUBFOLDER = "conditioning"


def _folder() -> str:
    path = os.path.join(folder_paths.get_output_directory(), SUBFOLDER)
    os.makedirs(path, exist_ok=True)
    return path


class MiniMaxH3XSaveConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XSaveConditioning",
            display_name="MiniMax H3 Save Conditioning",
            category="model/conditioning/minimax",
            search_aliases=[
                "h3", "minimax", "h3 image", "save conditioning",
                "cache prompt", "保存条件",
            ],
            description="Saves an encoded prompt to output/conditioning.",
            is_output_node=True,
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input("name", default="h3image"),
                io.Boolean.Input(
                    "overwrite",
                    default=False,
                    tooltip="Off: a taken name gets _2, _3, ...",
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Stored as a note.",
                ),
                io.String.Input(
                    "text_encoder",
                    default="",
                    optional=True,
                    tooltip="Stored as a note.",
                ),
            ],
            outputs=[io.String.Output(display_name="file")],
        )

    @classmethod
    def execute(
        cls, conditioning, name, overwrite, prompt="", text_encoder=""
    ) -> io.NodeOutput:
        folder = _folder()
        path = (
            os.path.join(folder, f"{clean_name(name)}.safetensors")
            if overwrite
            else free_path(folder, clean_name(name))
        )
        save_conditioning(
            path, conditioning, {"prompt": prompt, "text_encoder": text_encoder}
        )
        file = os.path.basename(path)
        return io.NodeOutput(
            file,
            ui=ui.PreviewText(
                f"saved {SUBFOLDER}/{file} ({os.path.getsize(path)} bytes)"
            ),
        )


class MiniMaxH3XLoadConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3XLoadConditioning",
            display_name="MiniMax H3 Load Conditioning",
            category="model/conditioning/minimax",
            search_aliases=[
                "h3", "minimax", "h3 image", "load conditioning", "加载条件",
            ],
            description="Loads a prompt saved by Save Conditioning.",
            inputs=[
                io.String.Input(
                    "name",
                    default="h3image",
                    tooltip="File name in output/conditioning.",
                )
            ],
            outputs=[io.Conditioning.Output(), io.String.Output(display_name="prompt")],
        )

    @classmethod
    def _path(cls, name: str) -> str:
        return os.path.join(_folder(), f"{clean_name(name)}.safetensors")

    @classmethod
    def fingerprint_inputs(cls, name):
        path = cls._path(name)
        if not os.path.exists(path):
            return name
        st = os.stat(path)
        return f"{st.st_mtime_ns}-{st.st_size}"

    @classmethod
    def validate_inputs(cls, name):
        if not os.path.exists(cls._path(name)):
            return f"No conditioning file {SUBFOLDER}/{clean_name(name)}.safetensors"
        return True

    @classmethod
    def execute(cls, name) -> io.NodeOutput:
        conditioning, info = load_conditioning(cls._path(name))
        return io.NodeOutput(conditioning, info.get("prompt", ""))
