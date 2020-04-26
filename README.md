# anki_card_generator

Takes a text file that includes the terms or words, their keywords, and their descriptions, and generates and downloads media for each.

It then generates a text file that, along with the media, can be imported into Anki as a set of notes.

Hopefully, this simplifies the workflow for creating new flashcards to study in Anki.

This tool uses another tool that I wrote: [`text2png`][text2png]. Just like that tool, this tool is meant solely for my use.

If you find a use for it, feel free to give attribution, but it's not required.

## Installation

This uses [Python](https://python.org).

Confirm `python` and `pip` are installed:

```sh
python --version
python -m pip --version
```

Install [`pipenv`](https://github.com/pypa/pipenv):

```sh
python -m pip install --user -U pipenv
```

[Download](https://github.com/mawillcockson/anki_card_generator/archive/master.zip) this repository, or [install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and then [clone this repository](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository).

The instructions for the latter option are platform specific, and the links contain information specific to macOS, Windows, and Linux. Android can use [Termux](https://termux.com/) to follow along with the Linux instructions.

Have `pipenv` install required packages:

```sh
pipenv install
```

Start a session with a `python` that has access to those packages:

```sh
pipenv shell --fancy
```

The above step is important, as running `pipenv run python` has caused some issues in the past, with files, upon being read, seeming to contain garbled data.

Show the help for this script:

```sh
python anki_card_generator.py -h
```

## Roadmap

Currently, this script is incomplete, and doesn't work.

Eventually, I'd like to be able to take characters and vocab, give them mnemonics, and have this script make files that are importable into Anki, complete with all the information I'd like to use them for studying.

Ideally, the implementation would make for easy customization, as different word sets might have different data sources, and different people may prefer to have different pieces of information in their flashcards.

Currently, I need to implement the following:

1. Querying online dictionaries for character and word information (e.g. Jisho.org, wiktionary)
  - [`zdict`](https://github.com/zdict/zdict) might fit the bill
1. Generate pictures using [`text2png`][text2png]
1. Make a data structure that reflects the way Anki shows notes/card information, to make it easier to modify what info should be exported
1. Generating decks that are importable into Anki
1. Manage Anki's media collection, to avoid re-importing the same files

## Copyright

See [`LICENSE`](./LICENSE).
