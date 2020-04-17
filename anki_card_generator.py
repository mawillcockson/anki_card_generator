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
    Dict,
)

from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
from invoke import run

## Program defaults
PROG_NAME = sys.argv[0]
default_media_dir = "~/scoop/persist/anki/data/User 1/collection.media"
default_log_level = logging.WARN

## Classes
Size = NamedTuple("Size", [("height", Number), ("width", Number)])
Position = NamedTuple("Position", [("x", Number), ("y", Number)])
SPath = Union[Path, str]
OptionalSPath = Optional[SPath]
Num = Union[int, float]


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


class Files(TypedDict):
    text: str
    picture: Path


class Pictures(Files):
    pass


class Notes(Files):
    pass


## Functions
def check_collisions(lines: List[str], directory: Union[Path, str]) -> Dict[str, Path]:
    dir_path = Path(directory)
    if not (dir_path.is_dir() and dir_path.exists()):
        raise ValueError(f"'{dir_path}' must be a directory")
    dir_contents = list(dir_path.iterdir())
    non_files = [str(path) for path in filter(lambda f: not f.is_file(), dir_contents)]
    colliding_names = [line for line in lines if (line + ".png") in non_files]
    for line in colliding_names:
        logging.error(
            f"'{line}.png' can't be created because there's something that's not a picture in the directory '{directory}' that already has that name"
        )
    if colliding_names:
        colliding_list = "\n".join(colliding_names)
        raise FileExistsError(
            f"These are names of files that would be created in '{directory}', but can't:\n{colliding_list}"
        )
    return {file.stem: file for file in dir_contents}


def get_notes(text_file: Union[str, Path]) -> List[NoteGroup]:
    if not Path(text_file).is_file():
        logging.error(f"'{text_file}' needs to be a file containing notes")

    text = Path(text_file).read_text()
    groups = text.split("\n\n")
    for group in groups:
        pass

    raise NotImplementedError(":(")


def generate_pictures(media_dir: SPath, picture_text: List[str]) -> Pictures:
    pass


def make_anki_notes(notes: List[NoteGroup], pictures: Pictures) -> Notes:
    pass


def main(
    text_file: SPath,
    media_dir: OptionalSPath = None,
    font: Optional[str] = None,
    size: Optional[Size] = None,
    padding: Optional[Num] = None,
    background: Optional[str] = None,
    text_color: Optional[str] = None,
    vocab: bool = False,
) -> None:
    notes_file = text_file

    if not media_dir:
        media_folder = Path(default_media_dir).expanduser().resolve()
    else:
        media_folder = Path(media_dir).expanduser().resolve()

    notes = get_notes(text_file=notes_file)
    picture_text = [note.word for note in notes if isinstance(note, VocabGroup)]
    picture_text.extend(
        [note.character for note in notes if isinstance(note, CharacterGroup)]
    )

    pictures = generate_pictures(media_dir=media_folder, picture_text=picture_text)
    make_anki_notes(notes=notes, pictures=pictures)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Uses a text file to generate and download media for Anki to import to make new notes"
    )

    def str_to_dir(path: str) -> Path:
        dir = Path(path)
        if dir.is_dir():
            return dir
        else:
            raise ArgumentTypeError(f"{dir} needs to be a directory that exists")

    def parse_log_level(level: str) -> int:
        levels = {
            "critical": logging.CRITICAL,
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }
        if not level.lower() in levels:
            raise ArgumentTypeError(
                f"'{level}' isn't one of: {' '.join(levels.keys())}"
            )
        return levels[level.lower()]

    parser.add_argument(
        "-f", "--file", type=Path, required=True, help="File with notes entries",
    )
    parser.add_argument(
        "-d",
        "--media-dir",
        type=str_to_dir,
        default=default_media_dir,
        help="Directory in which to output pictures",
    )
    parser.add_argument("--font", help="Font to use for text", required=False)
    parser.add_argument(
        "--size",
        help="Size in pixels to make all character images (e.g. 500x500)",
        required=False,
    )
    parser.add_argument(
        "--padding",
        type=float,
        help="The percentage of the canvas dimensions to use as a blank border",
        required=False,
    )
    parser.add_argument("--background", help="Color for the background", required=False)
    parser.add_argument(
        "--text-color", help="Color to use for the text", required=False
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        help="If passed, will overwrite existing files; otherwise, nothing is clobbered",
    )
    parser.add_argument(
        "--log",
        default=default_log_level,
        type=parse_log_level,
        help="Verbosity/log level",
    )

    args = parser.parse_args()
    try:
        main(
            text_file=args.file,
            media_dir=args.media_dir,
            font=args.font,
            size=args.size,
            padding=args.padding,
            background=args.background,
            text_color=args.text_color,
            vocab=args.vocab,
        )
    except Exception as err:
        parser.print_help()
