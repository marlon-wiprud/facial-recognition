import click
from service.service import train as exec_train
from service.service import recognize_face, introduce_webcam
from service.preprocessing import preprocess_data, augment_by_path, preview_imgs
from service.data_collection import run_webcam


@click.group()
@click.pass_context
def cli(ctx):
    # print('context: ', ctx)
    pass


@click.command()
def train():
    exec_train()


@click.command()
def preprocess():
    preprocess_data()


@click.command()
@click.option('--file', prompt='Image to recognize', help='The image to recognize')
def recognize():
    prediction = recognize_face()
    click.echo(prediction)


@click.command()
@click.option('--file', prompt='Image to augment', help='The image to augment')
def augment(file):
    _, aug_imgs = augment_by_path(file)
    preview_imgs(aug_imgs)


@click.command()
@click.option('--label', prompt='Name of person', help='Label name')
def introduce(label):
    introduce_webcam(label)


@click.command()
def who():
    run_webcam()


cli.add_command(train)
cli.add_command(preprocess)
cli.add_command(recognize)
cli.add_command(augment)
cli.add_command(introduce)
cli.add_command(who)
