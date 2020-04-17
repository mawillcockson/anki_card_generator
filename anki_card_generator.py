"""Uses a text file to generate and download media for Anki to import to make new notes"""
# mypy: ignore-missing-imports, follow-imports=silent, warn-unreachable, warn-unused-configs, disallow-any-generics
# mypy: disallow-subclassing-any, disallow-untyped-calls, disallow-untyped-defs, disallow-incomplete-defs
# mypy: check-untyped-defs, disallow-untyped-decorators, no-implicit-optional, warn-redundant-casts
# mypy: warn-unused-ignores, warn-return-any, no-implicit-reexport, strict-equality
import math
import sys
from numbers import Number
from pathlib import Path
from re import compile as re_compile
from typing import (
    Any,
    Callable,
    List,
    NamedTuple,
    NewType,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

from invoke import Collection, Config, Context, Program, task
from invoke.config import merge_dicts
from invoke.tasks import Task
from invoke.watchers import Responder

## Program defaults
# Can be overriden elsewhere, too
PROG_NAME = sys.argv[0]
default_font = "KanjiStrokeOrders"
default_output_dir = "./output"
default_canvas_size = "500x500"
default_padding = 10
default_background = "white"
default_text_color = "black"

## Classes used frequently
Size = NamedTuple("Size", [("height", Number), ("width", Number)])
Position = NamedTuple("Position", [("x", Number), ("y", Number)])
SPath = Union[Path, str]
OptionalSPath = Optional[SPath]


class CharacterGroup(NamedTuple):
    character: str
    keyword: str
    frame_number: Optional[int]
    story: str


class VocabGroup(NamedTuple):
    word: str
    definition: str
    example_sentences: List[str]
    story: str


NoteGroup = Union[CharacterGroup, VocabGroup]


class Pictures(TypedDict):
    text: str
    picture: Path


def get_notes(text_file: Union[str, Path]) -> List[NoteGroup]:
    pass


def generate_pictures(
    ctx: Context, media_dir: SPath, picture_text: List[str]
) -> Pictures:
    pass


@task( # type: ignore
    post=[],
    positional=[],  # Make all arguments into explicit flags by saying none of them are positional
    optional=["media-dir", "font", "size", "padding", "background", "text-color"],
)
def default_task(
    ctx: Context,
    text_file: OptionalSPath = None,
    media_dir: OptionalSPath = None,
    font: Optional[str] = None,
    size: Optional[str] = None,
    padding: Optional[int] = None,
    background: Optional[str] = None,
    text_color: Optional[str] = None,
) -> None:
    if not text_file:
        notes_file = ctx.args.text_file
    else:
        notes_file = text_file

    if not media_dir and not ctx.args.media_dir:
        media_folder = ctx.output.default
    elif not media_dir:
        media_folder = ctx.args.media_dir

    if not size:
        size = ctx.args.size

    if not font:
        font = ctx.args.font

    if padding == None:
        padding = ctx.args.padding

    if not background:
        background = ctx.args.background

    if not text_color:
        text_color = ctx.args.text_color

    notes = get_notes(text_file=notes_file)
    picture_text = [note.word for note in notes if isinstance(note, VocabGroup)]
    picture_text.extend(
        [note.character for note in notes if isinstance(note, CharacterGroup)]
    )

    pictures = generate_pictures(
        ctx=ctx, media_dir=media_folder, picture_text=picture_text
    )

    for group in notes:
        pass


if __name__ == "__main__":
    namespace = Collection()
    namespace.configure(
        {
            "output": {"default": default_output_dir},
            "font": {"default": default_font},
            "style": {
                "size": default_canvas_size,
                "padding": default_padding,
                "background": default_background,
                "text_color": default_text_color,
            },
        }
    )
    namespace.add_task(default_task, default=True)
    for i, name in enumerate(namespace.tasks["default-task"].post):
        namespace.tasks["default-task"].post[i] = task(name)
        namespace.add_task(task(name))

    class SetupConfig(Config):  # type: ignore
        prefix: str = PROG_NAME

        @staticmethod
        def global_defaults():  # type: ignore
            base_defaults = Config.global_defaults()
            overrides = {
                "tasks": {"collection_name": PROG_NAME},
            }
            return merge_dicts(base=base_defaults, updates=overrides)

    program = Program(
        name=PROG_NAME, namespace=namespace, config_class=SetupConfig, version="0.0.1"
    )
    program.run()
