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
from itertools import compress
from operator import not_
from csv import DictReader
from functools import partial

from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
from invoke import run

try:
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
        Iterator,
    )
except ImportError as err:
    assert sys.version_info >= (3, 8), "Requires Python 3.8 or higher"
    sys.exit("Could not import needed types from typing module")

## Program defaults
PROG_NAME = sys.argv[0]
default_media_dir = "~/scoop/persist/anki/data/User 1/collection.media"
default_log_level = logging.WARN
default_rtk_index_file = Path("../heisig_index/rtk_index.txt")
default_output_file = Path("./Anki_notes.csv")

## Classes
comment_re = re_compile(r"^\s*(#|＃).*$")
blank_re = re_compile(r"^\s+$")
first_word_re = re_compile(r"^\w+")
character_and_keyword_re = re_compile(r"^(?P<character>\w)\s+(?P<keyword>\w+)$")
character_index_re = re_compile(
    r"^(?P<index>\d+|\*)\s+(?P<character>\w)(\W+(?P<alternate>\w))?$"
)
story_re = re_compile(r"^story:\s+(?P<story>\w.*)$")
Size = NamedTuple("Size", [("height", Number), ("width", Number)])
Position = NamedTuple("Position", [("x", Number), ("y", Number)])
SPath = Union[Path, str]
OptionalSPath = Optional[SPath]
Num = Union[int, float]


class CharacterGroup(NamedTuple):
    character: str
    alternate: Optional[str]
    keyword: str
    frame_number: Optional[Union[int, str]]
    story: Optional[str]
    pronounciation: Optional[str]


HeisigLookup = Callable[[str], Optional[CharacterGroup]]


class VocabGroup(NamedTuple):
    word: str
    definition: str
    example_sentences: List[str]
    story: str


NoteGroup = Union[CharacterGroup, VocabGroup]


class Pictures(TypedDict):
    text: str
    picture: Path


## Functions
def error(message: str) -> Exception:
    logging.error(message.lstrip())
    return Exception(message.lstrip())


def comment_or_blank(string: str) -> bool:
    return bool(comment_re.search(string) or blank_re.search(string) or string == "")


def split_groups(paragraph: str) -> List[str]:
    assert isinstance(paragraph, str), f"'{paragraph}' is not a str"
    lines = paragraph.splitlines()
    current_paragraph: List[str] = list()
    paragraphs: List[str] = list()
    for line in lines:
        if comment_or_blank(line):
            continue
        if line == "":
            paragraphs.append("\n".join(current_paragraph))
        else:
            current_paragraph.append(line)
    if len(current_paragraph) > 0:
        paragraphs.append("\n".join(current_paragraph))
    return paragraphs


def which_exist(names: List[str], directory: Union[Path, str]) -> Dict[str, Path]:
    """Checks which file names are already taken in a directory,
    and raises an error if it wasn't a file that took the name"""
    dir_path = Path(directory)
    if not (dir_path.is_dir() and dir_path.exists()):
        raise ValueError(f"'{dir_path}' must be a directory")
    dir_contents = list(dir_path.iterdir())
    non_files = [str(path) for path in filter(lambda f: not f.is_file(), dir_contents)]
    colliding_names = [name for name in names if name in non_files]
    for name in colliding_names:
        logging.error(
            f"'{name}.png' can't be created because there's something that's not a picture in the directory '{directory}' that already has that name"
        )
    if colliding_names:
        colliding_list = "\n".join(colliding_names)
        raise FileExistsError(
            f"These are names of files that would be created in '{directory}', but can't:\n{colliding_list}"
        )
    return {file.name: file for file in dir_contents}


# I was thinking these could be arpeggio-based parsers, just for fun
def parse_vocab_group(paragraph: str) -> VocabGroup:
    assert isinstance(paragraph, str), f"'{paragraph}' is not str"
    lines = paragraph.splitlines()
    story_lines = list(map(story_re.search, lines))
    story_matches = list(filter(None, story_lines))
    non_stories = list(compress(lines, map(not_, story_lines)))
    if len(non_stories) < 3:
        raise error(
            f"""Vocab group has word on line 1, definition on line 2, and 1 or more
                example sentences on lines following, with an optional story/mnemonic
                indicated by a line starting with 'story: '
                Does not match:
                {paragraph}"""
        )
    if len(story_matches) > 0:
        story = story_matches[0].group(0)
    else:
        story = ""
    return VocabGroup(
        word=non_stories[0],
        definition=non_stories[1],
        example_sentences=non_stories[2:],
        story=story,
    )


def rtk_index_gen(rtk_index_file: SPath) -> HeisigLookup:
    """Expects a file to be present in the same directory, where each line
    contains the frame number and corresponding character from Remembering the Kanji vol1, v6"""
    rtk_index = Path(rtk_index_file)
    if not rtk_index.is_file():
        raise error(f"Expected to find file '{rtk_index}'")
    lookup_dict: Dict[str, CharacterGroup] = dict()
    with rtk_index.open() as f:
        dreader = DictReader(
            f,
            fieldnames=[
                "character",
                "index",
                "keyword",
                "alternate",
                "pronounciation",
                "story",
            ],
        )
        for row in dreader:
            lookup_dict[row["character"]] = CharacterGroup(**row)

    def lookup(character: str) -> Optional[CharacterGroup]:
        return lookup_dict.get(character)

    return lookup


def parse_character_group(
    paragraph: str, heisig_lookup: HeisigLookup,
) -> CharacterGroup:
    assert isinstance(paragraph, str), f"'{paragraph}' is not a str"
    lines = paragraph.splitlines()
    if len(lines) < 2:
        raise error(
            """character groups must have at least 2 lines:
               the character on the first line,
               and the story or mnemonic on the following lines"""
        )

    character = lines[0]
    heisig_info: Optional[CharacterGroup] = heisig_lookup(character)
    if not heisig_info:
        raise ValueError(f"Can't find '{character}' in Heisig index")
    story = "\n".join(lines[1:])
    return CharacterGroup(
        character=character,
        keyword=heisig_info.keyword,
        frame_number=heisig_info.frame_number,
        story=story,
        pronounciation=heisig_info.pronounciation,
        alternate=heisig_info.alternate,
    )


def is_character_group(paragraph: str, heisig_lookup: HeisigLookup) -> bool:
    match = first_word_re.search(paragraph)
    return bool(heisig_lookup(match.group() if match else ""))


def get_notes(
    text_file: Union[str, Path], heisig_lookup: HeisigLookup
) -> List[NoteGroup]:
    if not Path(text_file).is_file():
        raise error(f"'{text_file}' needs to be a file containing notes")

    text = Path(text_file).read_text()
    groups = split_groups(text)
    parse_characters = partial(parse_character_group, heisig_lookup=heisig_lookup)
    is_char_group = partial(is_character_group, heisig_lookup=heisig_lookup)
    is_not_char_group = lambda g: not is_char_group(g)
    characters = map(parse_characters, filter(is_char_group, groups))
    vocab = map(parse_vocab_group, filter(is_not_char_group, groups))
    notes: List[NoteGroup] = list(characters)
    notes.extend(vocab)
    return notes


def generate_pictures(
    media_dir: SPath, picture_text: List[str], clobber: bool
) -> Pictures:
    raise NotImplementedError("Sorry, can't generate pictures yet")


def make_anki_notes(
    notes: List[NoteGroup], pictures: Pictures, output_csv: SPath
) -> None:
    raise NotImplementedError("Sorry, can't generate notes yet")


def main(
    text_file: SPath,
    heisig_lookup: HeisigLookup,
    media_dir: OptionalSPath = default_media_dir,
    output_file: SPath = default_output_file,
    clobber: bool = False,
    font: Optional[str] = None,
    size: Optional[Size] = None,
    padding: Optional[Num] = None,
    background: Optional[str] = None,
    text_color: Optional[str] = None,
) -> None:
    notes_file = text_file

    if not media_dir:
        media_folder = Path(default_media_dir).expanduser().resolve()
    else:
        media_folder = Path(media_dir).expanduser().resolve()

    output_csv = Path(output_file).expanduser().resolve()

    notes = get_notes(text_file=notes_file, heisig_lookup=heisig_lookup)
    picture_text = [note.word for note in notes if isinstance(note, VocabGroup)]
    picture_text.extend(
        [note.character for note in notes if isinstance(note, CharacterGroup)]
    )

    pictures = generate_pictures(
        media_dir=media_folder, picture_text=picture_text, clobber=clobber
    )
    make_anki_notes(notes=notes, pictures=pictures, output_csv=output_csv)


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
            "fatal": logging.FATAL,
            "critical": logging.CRITICAL,
            "error": logging.ERROR,
            "warn": logging.WARN,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }
        if not level.lower() in levels:
            raise ArgumentTypeError(
                f"'{level}' isn't one of: {' '.join(levels.keys())}"
            )
        return levels[level.lower()]

    def path_to_heisiglookup(path: SPath) -> HeisigLookup:
        index_path = Path(path)
        try:
            assert index_path.stat().st_size > 0
        except FileNotFoundError as err:
            raise ArgumentTypeError(f"Index file '{index_path}' not found")
        except AssertionError as err:
            raise ArgumentTypeError(f"Empty index file: '{index_path}'")
        if not index_path.is_file():
            raise ArgumentTypeError(f"Not a file: '{index_path}'")
        return rtk_index_gen(index_path)

    parser.add_argument(
        "-f", "--file", type=Path, required=True, help="File with notes entries",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=path_to_heisiglookup,
        help=f"CSV file with information from Heisig's RTK vol1, 6th edition; default '{default_rtk_index_file}'",
    )
    parser.add_argument(
        "-d",
        "--media-dir",
        type=str_to_dir,
        default=default_media_dir,
        help=f"Directory in which to output pictures; default '{default_media_dir}'",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=default_output_file,
        help=f"File to write Anki notes to; default '{default_output_file}'",
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
        help=f"Verbosity/log level; default '{logging.getLevelName(default_log_level)}'",
    )

    args = parser.parse_args()
    if not args.index:
        heisig_index = path_to_heisiglookup(default_rtk_index_file)
    else:
        heisig_index = args.index
    try:
        main(
            text_file=args.file,
            media_dir=args.media_dir,
            output_file=args.output_file,
            clobber=args.clobber,
            font=args.font,
            size=args.size,
            padding=args.padding,
            background=args.background,
            text_color=args.text_color,
            heisig_lookup=heisig_index,
        )
    except Exception as err:
        parser.print_help()
