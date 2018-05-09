#!/usr/bin/env python
import csv
from pathlib import Path
from typing import List
from typing import Union

import click

from commands.combine import combine_strain_files
from commands.crop import crop_files
from commands.rename import rename_plate_files
from commands.types import StrainInfo


class Config:
    _home_directory: Path
    plates_group: str

    @property
    def home_directory(self) -> Path:
        return self._home_directory

    @home_directory.setter
    def home_directory(self, value: str) -> None:
        self._home_directory = Path(value)

    @property
    def raw_plates_directory(self) -> Path:
        return self._home_directory / self.plates_group

    @property
    def cropped_directory(self) -> Path:
        return self._prepare_directory('cropped')

    @property
    def combined_directory(self) -> Path:
        return self._prepare_directory('combined')

    def _prepare_directory(self, suffix: str) -> Path:
        directory = self._home_directory / (self.plates_group + '-' + suffix)
        directory.mkdir(exist_ok=True)
        return directory


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('home',
                type=click.Path(exists=True, file_okay=False, writable=True, resolve_path=True))
@click.argument('plates_group', type=str)
@pass_config
def main(config: Config, home: click.Path, plates_group: str) -> None:
    config.home_directory = home
    config.plates_group = plates_group


@main.command()
@click.option('-d', '--description_file', type=click.File())
@click.option('-e', '--extensions', default='jpg,arw')
@pass_config
def rename(config: Config, description_file: click.File(lazy=True), extensions: str) -> None:
    extension = extensions.split(',')[0]
    rename_plate_files(
        images_folder=config.raw_plates_directory,
        strains=get_names(description_file.name),
        extension=prepare_extension(extension),
    )


@main.command()
@click.option('-e', '--extensions', default='jpg')
@pass_config
def crop(config: Config, extensions: str) -> None:
    extension = extensions.split(',')[0]
    crop_files(
        extension=prepare_extension(extension),
        input_folder=config.raw_plates_directory,
        target_folder=config.cropped_directory,
    )


@main.command()
@click.option('-e', '--extensions', default='jpg')
@click.option('-r', '--rotate', is_flag=True)
@pass_config
def combine(config: Config, extensions: str, rotate: bool) -> None:
    extension = extensions.split(',')[0]
    combine_strain_files(
        extension=prepare_extension(extension),
        input_folder=config.cropped_directory,
        target_folder=config.combined_directory,
        with_rotation=rotate,
    )


def prepare_extension(extension: str) -> str:
    extension = extension.lower()
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    return extension


def get_names(strain_descriptions_file: Union[Path, str]) -> List[StrainInfo]:
    with open(strain_descriptions_file, newline='') as csv_file:
        next(csv_file, None)

        return [StrainInfo(name=record[1], plates_number=int(record[2]))
                for record in csv.reader(csv_file, dialect='excel')]


if __name__ == '__main__':
    main()
