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

## Classes
comment_re = re_compile(r"^\s*(#|＃).*$")
blank_re = re_compile(r"^\s+$")
ends_with_keyword_re = re_compile(r"^[\w\s]+[ a-zA-Z-]+$")
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
FrameLookup = Callable[[str], Union[int, str, None]]


class CharacterGroup(NamedTuple):
    character: str
    alternate: Optional[str]
    keyword: str
    frame_number: Optional[Union[int, str]]
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
def error(message: str) -> Exception:
    logging.error(message.lstrip())
    return Exception(message.lstrip())


def split_groups(paragraph: str) -> List[str]:
    assert isinstance(paragraph, str), f"'{paragraph}' is not a str"
    lines = paragraph.splitlines()
    current_paragraph: List[str] = list()
    paragraphs: List[str] = list()
    for line in lines:
        if line == "":
            paragraphs.append("\n".join(current_paragraph))
        else:
            current_paragraph.append(line)
    if len(current_paragraph) > 0:
        paragraphs.append("\n".join(current_paragraph))
    return paragraphs


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


def not_comment_or_blank(string: str) -> bool:
    return not (comment_re.search(string) or blank_re.search(string) or string == "")


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


def rtk_index_gen(rtk_index_file: SPath) -> FrameLookup:
    """Expects a file to be present in the same directory, where each line
    contains the frame number and corresponding character from Remembering the Kanji vol1, v6"""
    rtk_index = Path(rtk_index_file)
    if not rtk_index.is_file():
        raise error(f"Expected to find file '{rtk_index}'")
    lines = rtk_index.read_text().splitlines()
    match_list = list(map(character_index_re.search, lines))
    non_matches = list(compress(lines, map(lambda x: x == None, match_list)))
    for i, non_match in enumerate(non_matches):
        logging.error(
            f"line {i} does not match pattern of number or '*', a space, and a character: {non_match}"
        )
    if len(non_matches) > 0:
        raise ValueError(f"Does not match format:\n{non_matches}")

    matches = filter(None, match_list)
    lookup_dict: Dict[str, Union[int, str]] = dict()
    for match in matches:
        index, character = match.groups()
        if index.isdigit():  # Recognizes full-width characters
            lookup_dict[character] = int(index)
        else:
            lookup_dict[character] = "*"

    def lookup(character: str) -> Union[int, str, None]:
        return lookup_dict.get(character)

    return lookup


def parse_character_group(
    paragraph: str, character_to_frame_number: FrameLookup
) -> CharacterGroup:
    assert isinstance(paragraph, str), f"'{paragraph}' is not a str"
    lines = paragraph.splitlines()
    if not len(lines) == 2:
        raise error(
            """character groups must have exactly 2 lines:
               the character and keyword on the first line separated by a space,
               and the story or mnemonic on the second line"""
        )

    character_and_keyword = lines[0]
    match = character_and_keyword_re.search(character_and_keyword)
    if not match:
        raise error(f"The following is not a character group:\n{paragraph}")
    character, keyword = match.groups()
    frame_number = character_to_frame_number(character)
    story = "\n".join(lines[1:])
    return CharacterGroup(
        character=character, keyword=keyword, frame_number=frame_number, story=story
    )


def is_character_group(paragraph: str) -> bool:
    return bool(ends_with_keyword_re.search(paragraph))


def get_notes(
    text_file: Union[str, Path], frame_lookup: FrameLookup
) -> List[NoteGroup]:
    if not Path(text_file).is_file():
        raise error(f"'{text_file}' needs to be a file containing notes")

    text = Path(text_file).read_text()
    groups = split_groups(text)
    filter_comments = lambda par: "\n".join(
        filter(not_comment_or_blank, par.split("\n"))
    )
    filtered_groups = map(filter_comments, groups)
    filter_characters = (
        lambda par: parse_character_group(par, frame_lookup)
        if is_character_group(par)
        else par
    )
    characters_filtered = map(filter_characters, filtered_groups)
    filter_vocab = (
        lambda par: parse_vocab_group(par) if is_character_group(par) else par
    )
    vocab_filtered = map(filter_vocab, characters_filtered)
    return list(vocab_filtered)


def generate_pictures(media_dir: SPath, picture_text: List[str]) -> Pictures:
    pass


def make_anki_notes(notes: List[NoteGroup], pictures: Pictures) -> Notes:
    pass


def main(
    text_file: SPath,
    frame_lookup: FrameLookup,
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

    notes = get_notes(text_file=notes_file, frame_lookup=frame_lookup)
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

    def path_to_framelookup(path: SPath) -> FrameLookup:
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
        type=path_to_framelookup,
        help="File with heisig frame numbers and their corresponding character(s)",
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
    if not args.index:
        heisig_index = path_to_framelookup(default_rtk_index_file)
    else:
        heisig_index = args.index
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
            frame_lookup=heisig_index,
        )
    except Exception as err:
        parser.print_help()
