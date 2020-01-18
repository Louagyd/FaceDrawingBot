import logging
import numpy as np
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Updater, Filters
import cv2
# import matplotlib.pyplot as plt
# print("importing pix2pix")
#import pix2pixtensorflow.pix2pix as p2p
import tensorflow as tf

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def restore_graph(checkpoint, seed = None):

    # inputs and targets are [batch_size, height, width, channels]

    tf.train.import_meta_graph("./model/export.meta")
    saver = tf.train.Saver(max_to_keep=1)
    sess = tf.InteractiveSession()
    print("loading model from checkpoint")
    checkpt = tf.train.latest_checkpoint(checkpoint)
    saver.restore(sess, checkpt)
    inputs = tf.get_default_graph().get_tensor_by_name("InTensor:0")
    outputs = tf.get_default_graph().get_tensor_by_name("OUTT:0")
    return sess, inputs, outputs

def generate_single_image(sess, path, input_ph, output_ph):
    imm = cv2.imread(path)/255
    imm_r = cv2.resize(imm, (256, 256))
    fd = {input_ph:imm_r}
    outs = sess.run(output_ph, feed_dict = fd)[0]

    # plt.imshow(outs), plt.show()
    return outs


the_sess, ins, outs = restore_graph(checkpoint="model")
#ots = generate_single_image(the_sess, "mmm.png", ins, outs)
# p2p.export_model("pix2pixtensorflow/faces2")
print("done")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
TOKEN = '965026452:AAHhv1WGGu4vaRNeWjiWS3qiDBWc0d4Iqrw'

def get_photo(bot, update):
    chat_id = update.message.chat_id
    print(chat_id)
    update.message.reply_text("🌬👻👻👻👻")
    file_id = update.message.photo[-1]
    newFile = bot.getFile(file_id)
    newFile.download("temp.jpg")
    image_r = cv2.imread('temp.jpg')
    grayImage = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
    if np.mean(grayImage) > 127:
        thresh = np.mean(grayImage)*0.8
    else:
        thresh = np.mean(grayImage)*1.3
    (threshold, imagebw) = cv2.threshold(grayImage, thresh, 255, cv2.THRESH_BINARY)
    if np.sum(imagebw == 0) < np.sum(imagebw == 255):
        imagebw = 255 - imagebw

    # plt.imshow(np.asarray([imagebw]*3).transpose([1,2,0])), plt.show()

    resized_image = cv2.resize(imagebw, (178, 218))
    # edges_image = funcs.detect_edges(resized_image)
    edges_image = resized_image
    # update.message.reply_text("saving")
    cv2.imwrite('temp2.jpg', edges_image)
    # bot.send_photo(chat_id, photo=open('temp2.jpg', 'rb'))
    # update.message.reply_text("gening")
    sageev = generate_single_image(the_sess, path="temp2.jpg", input_ph=ins, output_ph=outs)
    sageev = cv2.cvtColor(sageev, cv2.COLOR_BGR2RGB)
    cv2.imwrite("temp3.jpg", sageev)
    bot.send_photo(chat_id, photo=open('temp3.jpg', 'rb'))

    # print(sageev)


def start(bot, update):
    update.message.reply_text("hiiiii, be khafan tarin bote donya khosh umadi. age balad nisti in bot chetori kaar mikone, az yeki ke balade bepors😈")

def setup():
    logging.basicConfig(level=logging.WARNING)

    updater = Updater(TOKEN)
    bot = updater.bot
    dp = updater.dispatcher

    photo_hd = MessageHandler(Filters.photo, get_photo)
    dp.add_handler(photo_hd)
    dp.add_handler(CommandHandler("start", start))

    bot.set_webhook()  # Delete webhook
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    setup()


