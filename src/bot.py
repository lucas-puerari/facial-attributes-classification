import os
import io
import yaml
import datetime
import numpy as np
from PIL import Image
from functools import partial
import tensorflow as tf
from dotenv import load_dotenv

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


from utils.pipeline import preprocessing, normalization
from utils.plotter import plot_dataset_image


# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def photo(update: Update, context: CallbackContext, model, config) -> None:
    file = context.bot.get_file(update.message.photo[-1].file_id)
    file = io.BytesIO(file.download_as_bytearray())

    image = Image.open(file)
    image = np.asarray(image).astype('float32')

    dataset = tf.data.Dataset.from_tensor_slices([image])

    # Preprocessing
    dataset = dataset.map(
        lambda image:
            tf.py_function(
                preprocessing, 
                [image, config['dataset']['image']['size'], ''],
                [tf.float32, tf.string]
            ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # Normalization
    dataset = dataset.map(
        lambda image, label: tf.py_function(normalization, [image, label], [tf.float32, tf.string]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    dataset = dataset.batch(config['training']['batch_size'])

    # Inference
    images, _ = next(iter(dataset))

    filepath = os.path.join(
        os.getcwd(), 
        config['project']['predictions'], 
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    plot_dataset_image(images[0], filepath)

    predictions = model(images, training=False)
    prediction = predictions.numpy()[0]
    prediction = np.rint(prediction).astype(dtype=bool)

    text = [f'{tuple[0]}: {tuple[1]}' for tuple in list(zip(config['dataset']['labels'], prediction))]
    text = str('\n'.join(text))
    
    context.bot.sendMessage(chat_id=update.message.chat_id, text=text)


def main() -> None:
    """Start the bot."""

    load_dotenv()

    MODEL_NAME = os.getenv('MODEL_NAME')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    ROOT_DIR = os.getcwd()
    CONFIG_FILE = os.path.join(ROOT_DIR, 'models', MODEL_NAME, 'config.yml')

    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_path = os.path.join(ROOT_DIR, config['project']['models'], MODEL_NAME)

    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Create the Updater and pass it your bot's token.
    updater = Updater(TELEGRAM_BOT_TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.photo, partial(photo, model=model, config=config)))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
