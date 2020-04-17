# anki_card_generator

Takes a text file that includes the terms or words, their keywords, and their descriptions, and generates and downloads media for each.

It then generates a text file that, along with the media, can be imported into Anki.

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

## Copyright

See [`LICENSE`](./LICENSE).
