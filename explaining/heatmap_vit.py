import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

patch_size = 16  # Size of the patches to be extract from the input images
projection_dim = 64
num_heads = 12
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 12
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
RESOLUTION = 128

crop_layer = keras.layers.CenterCrop(RESOLUTION, RESOLUTION)
norm_layer = keras.layers.Normalization(
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
)
rescale_layer = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)

def get_img_array(img_path, size):
    # `img` is a PIL image of size 128x128
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 128, 128, 3)
    # array = np.expand_dims(array, axis=0)
    return array

def preprocess_image(image, model_type, size=RESOLUTION):
    # Turn the image into a numpy array and add batch dim.
    image = np.array(image)
    image = tf.expand_dims(image, 0)

    # If model type is vit rescale the image to [-1, 1].
    if model_type == "original_vit":
        image = rescale_layer(image)

    # Resize the image using bicubic interpolation.
    resize_size = int((256 / 224) * size)
    image = tf.image.resize(image, (resize_size, resize_size), method="bicubic")

    # Crop the image.
    image = crop_layer(image)

    # If model type is DeiT or DINO normalize the image.
    if model_type != "original_vit":
        image = norm_layer(image)

    return image.numpy()

def attention_heatmap(attention_score_dict, image, model_type="dino"):
    num_tokens = 2 if "distilled" in model_type else 1

    # Sort the Transformer blocks in order of their depth.
    attention_score_list = list(attention_score_dict.keys())
    attention_score_list.sort(key=lambda x: int(x.split("_")[-2]), reverse=True)

    # Process the attention maps for overlay.
    w_featmap = image.shape[2] // patch_size
    h_featmap = image.shape[1] // patch_size
    attention_scores = attention_score_dict[attention_score_list[0]]

    # Taking the representations from CLS token.
    attentions = attention_scores[0, :, 0, num_tokens:].reshape(num_heads, -1)

    # Reshape the attention scores to resemble mini patches.
    attentions = attentions.reshape(num_heads, w_featmap, h_featmap)
    attentions = attentions.transpose((1, 2, 0))

    # Resize the attention patches to 224x224 (224: 14x16).
    attentions = tf.image.resize(
        attentions, size=(h_featmap * patch_size, w_featmap * patch_size)
    )
    return attentions

if __name__ == "__main__":    
    model = keras.models.load_model('/home/smachmeier/results/models/save_at_40_binary_vit_flow-minp2-dim16-cols8-ALL-NONE-split-ratio', compile=False)
    inputs = keras.Input((RESOLUTION, RESOLUTION, 3))
    outputs, attention_weights = model(inputs, training=False)

    model_2 = keras.Model(inputs, outputs=[outputs, attention_weights])

    img = get_img_array("/home/smachmeier/data/binary-flow-minp2-dim16-cols8-ALL-NONE/malware/Htbot-5768.pcap_processed.png", (128,128))
    # img = get_img_array("/home/smachmeier/data/binary-flow-minp2-dim16-cols8-ALL-NONE/malware/Virut-2314.pcap_processed.png", (128,128))

    preprocessed_image = preprocess_image(img, "original_vit")
    # preprocessed_image = img
   
    predictions, attention_score_dict = model_2.predict(
        preprocessed_image
    )
    print(predictions)

    # De-normalize the image for visual clarity.
    in1k_mean = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])
    in1k_std = tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])
    preprocessed_img_orig = (preprocessed_image * in1k_std) + in1k_mean
    preprocessed_img_orig = preprocessed_img_orig / 255.0
    preprocessed_img_orig = tf.clip_by_value(preprocessed_img_orig, 0.0, 1.0).numpy()

    # Generate the attention heatmaps.
    attentions = attention_heatmap(attention_score_dict, preprocessed_img_orig)

    # img = get_img_array("/home/smachmeier/data/binary-flow-minp2-dim16-cols8-ALL-NONE/malware/Htbot-5768.pcap_processed.png", (128,128))
    img2 = get_img_array("/home/smachmeier/data/binary-flow-minp2-dim16-cols8-ALL-NONE/malware/Virut-2314.pcap_processed.png", (128,128))

    preprocessed_image = preprocess_image(img, "original_vit")
    # preprocessed_image = img
   
    predictions, attention_score_dict = model_2.predict(
        preprocessed_image
    )
    print(predictions)

    # De-normalize the image for visual clarity.
    in1k_mean = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])
    in1k_std = tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])
    preprocessed_img_orig = (preprocessed_image * in1k_std) + in1k_mean
    preprocessed_img_orig = preprocessed_img_orig / 255.0
    preprocessed_img_orig = tf.clip_by_value(preprocessed_img_orig, 0.0, 1.0).numpy()

    # Generate the attention heatmaps.
    attentions2 = attention_heatmap(attention_score_dict, preprocessed_img_orig)

    assert(np.array_equal(attentions[..., 11], attentions2[..., 11]))

    # Plot the maps.
    # fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(13, 13))
    # img_count = 0

    # for i in range(3):
    #     for j in range(4):
    #         if img_count < len(attentions):
    #             axes[i, j].imshow(preprocessed_img_orig[0])
    #             axes[i, j].imshow(attentions[..., img_count], cmap="inferno", alpha=0.6)
    #             axes[i, j].title.set_text(f"Attention head: {img_count}")
    #             axes[i, j].axis("off")
    #             img_count += 1
    # plt.show()
    # plt.tight_layout()
    # plt.savefig("attention-heatmaps.jpg")
    # plt.imshow(preprocessed_img_orig[0])
    # plt.imshow(attentions[..., 11], cmap="inferno", alpha=0.6)
    # plt.axis("off")
    # plt.show()
    # plt.tight_layout()
    # plt.savefig("attention-heatmaps-Htbot-5768.pdf")
