import glob
import hashlib
import os

import click
import splitfolders


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


_split_options = [
    click.option(
        "-w",
        "--write",
        "output_dir",
        type=click.Path(),
        required=True,
        help="Destination file path, stores result.",
    ),
    click.option(
        "--remove-duplicates",
        "remove_duplicates",
        is_flag=True,
        default=False,
        help="Within a single output folder belonging to a single input folder no duplicate images will be produced if two inputs lead to the same image.",
    ),
    click.option("-r", "--read", "input_dir", required=True, type=click.Path()),
]


@click.group(name="split", context_settings={"show_default": True})
def split():
    click.secho("Split PCAP Images to dataset")


@split.command(name="fixed")
@add_options(_split_options)
def split_fixed(output_dir, input_dir, remove_duplicates):
    splitfolders.fixed(
        input_dir,
        output=output_dir,
        fixed=(100000, 25000, 10000),
        seed=1337,
    )
    if remove_duplicates:
        remove_duplicates(output_dir)


@split.command(name="ratio")
@add_options(_split_options)
def split_ratio(output_dir, input_dir, remove_duplicates):
    splitfolders.ratio(
        input_dir,
        output=output_dir,
        seed=1337,
        ratio=(0.7, 0.2, 0.1),
    )
    if remove_duplicates:
        remove_duplicates(output_dir)


def remove_duplicates(directory):
    hashes = set()
    hashes_file = {}
    for filename in glob.iglob(directory + "/*/**/*.png", recursive=True):
        digest = hashlib.sha1(open(filename, "rb").read()).digest()
        if digest not in hashes:
            hashes.add(digest)
            hashes_file[digest] = filename
        else:
            os.remove(hashes_file.get(digest))
            os.remove(path)


if __name__ == "__main__":
    split()
