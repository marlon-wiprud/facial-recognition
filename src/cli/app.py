import click
from service.service import train as exec_train
from service.service import recognize_face
from service.preprocessing import preprocess_data


@click.group()
@click.pass_context
def cli(ctx):
    print('context: ', ctx)
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


cli.add_command(train)
cli.add_command(preprocess)
cli.add_command(recognize)
