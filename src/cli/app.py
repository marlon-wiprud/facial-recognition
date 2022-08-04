import click
from service.service import train as exec_train
from service.service import recognize_face, introduce_webcam, run_webcam
from service.preprocessing import preprocess_data_from_folder, augment_by_path, preview_imgs
from service.utils import get_latest_model_folder


@click.group()
def cli():
    pass


@click.command()
def train():
    exec_train()


@click.command()
def preprocess():
    preprocess_data_from_folder()


@click.command()
@click.option('--file', prompt='Image to recognize', help='The image to recognize')
def recognize():
    prediction = recognize_face()
    click.echo(prediction)


@click.command()
@click.option('--file', prompt='Image to augment', help='The image to augment')
def augment_preview(file):
    _, aug_imgs = augment_by_path(file)
    preview_imgs(aug_imgs)


@click.command()
@click.option('--label', prompt='Name of person', help='Label name')
def introduce(label):
    introduce_webcam(label)


@click.command()
def webcam():
    run_webcam()


@click.command()
def latest_model_version():
    folder = get_latest_model_folder()
    print('latest model version: ', folder)


cli.add_command(train)
cli.add_command(preprocess)
cli.add_command(recognize)
cli.add_command(augment_preview)
cli.add_command(introduce)
cli.add_command(webcam)
cli.add_command(latest_model_version)
