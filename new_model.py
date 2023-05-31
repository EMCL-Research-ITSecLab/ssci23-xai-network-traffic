import ml_collections
import math
from typing import List, Tuple

import tensorflow as tf
from keras import layers
from ml_collections import ConfigDict
from tensorflow import keras

def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.batch_size = 32
    config.buffer_size = config.batch_size * 2
    config.input_shape = (128, 128, 3)

    config.image_size = 128
    config.patch_size = 16
    config.num_patches = (config.image_size // config.patch_size) ** 2
    config.num_classes = 2

    config.pos_emb_mode = "sincos"

    config.initializer_range = 0.02
    config.layer_norm_eps = 1e-6
    config.projection_dim = 768
    config.num_heads = 12
    config.num_layers = 12
    config.mlp_units = [
        config.projection_dim * 4,
        config.projection_dim,
    ]
    config.dropout_rate = 0.0
    config.classifier = "token"

    return config.lock()

class TFViTSelfAttention(keras.layers.Layer):
    def __init__(self, config: ConfigDict, **kwargs):
        super().__init__(**kwargs)

        if config.projection_dim % config.num_heads != 0:
            raise ValueError(
                f"The hidden size ({config.projection_dim}) is not a multiple of the number "
                f"of attention heads ({config.num_heads})"
            )

        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.projection_dim / config.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            name="query",
        )
        self.key = keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            name="key",
        )
        self.value = keras.layers.Dense(
            units=self.all_head_size,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            name="value",
        )
        self.dropout = keras.layers.Dropout(rate=config.dropout_rate)

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(
            tensor=tensor,
            shape=(
                batch_size,
                -1,
                self.num_attention_heads,
                self.attention_head_size,
            ),
        )

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = tf.shape(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)
        mixed_key_layer = self.key(inputs=hidden_states)
        mixed_value_layer = self.value(inputs=hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(inputs=attention_probs, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)

        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(
            tensor=attention_output, shape=(batch_size, -1, self.all_head_size)
        )
        outputs = (
            (attention_output, attention_probs)
            if output_attentions
            else (attention_output,)
        )

        return outputs


class TFViTSelfOutput(keras.layers.Layer):
    """
    The residual connection is defined in TFViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ConfigDict, **kwargs):
        super().__init__(**kwargs)

        self.dense = keras.layers.Dense(
            units=config.projection_dim,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=config.initializer_range
            ),
            name="dense",
        )
        self.dropout = keras.layers.Dropout(rate=config.dropout_rate)

    def call(
        self,
        hidden_states: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)

        return hidden_states


class TFViTAttention(keras.layers.Layer):
    def __init__(self, config: ConfigDict, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFViTSelfAttention(config, name="attention")
        self.dense_output = TFViTSelfOutput(config, name="output")

    def call(
        self,
        input_tensor: tf.Tensor,
        head_mask: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            head_mask=head_mask,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self.dense_output(
            hidden_states=self_outputs[0] if output_attentions else self_outputs,
            training=training,
        )
        if output_attentions:
            outputs = (attention_output,) + self_outputs[
                1:
            ]  # add attentions if we output them

        return outputs


class PositionalEmbedding(layers.Layer):
    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Compute the positions.
        positions = self.config.num_patches
        positions += 1 if self.config.classifier == "token" else 0

        # Build the sequence of positions in 1D.
        self.pos_flat_patches = tf.range(positions, dtype=tf.float32, delta=1)

        # Encode the positions with an Embedding layer.
        if self.config.pos_emb_mode == "learn":
            self.pos_embedding = layers.Embedding(
                input_dim=self.config.num_patches + 1
                if self.config.classifier == "token"
                else self.config.num_patches,
                output_dim=self.config.projection_dim,
                embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
            )

    def get_config(self):
        config = super().get_config()
        config.update(self.config)
        return config

    def get_1d_sincos_pos_embed(self):
        # Inspired from https://github.com/huggingface/transformers/blob/master/src/transformers/models/vit_mae/modeling_vit_mae.py#L184.
        # Build the sine-cosine positional embedding.
        omega = tf.range(self.config.projection_dim // 2, dtype=tf.float32)
        omega /= self.config.projection_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        out = tf.einsum(
            "m,d->md", self.pos_flat_patches, omega
        )  # (M, D/2), outer product

        emb_sin = tf.sin(out)  # (M, D/2)
        emb_cos = tf.cos(out)  # (M, D/2)

        emb = tf.concat([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def get_learnable_pos_embed(self):
        emb = self.pos_embedding(self.pos_flat_patches)
        return emb

    def call(self, inputs):
        if self.config.pos_emb_mode == "learn":
            pos_emb = self.get_learnable_pos_embed()
        else:
            pos_emb = self.get_1d_sincos_pos_embed()

        # Inject the positional embeddings with the tokens.
        if pos_emb.dtype != inputs.dtype:
            pos_emb = tf.cast(pos_emb, inputs.dtype)
        outputs = inputs + pos_emb
        return outputs


def mlp(x: int, dropout_rate: float, hidden_units: List):
    """FFN for a Transformer block."""
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for idx, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            activation=tf.nn.gelu if idx == 0 else None,
            kernel_initializer="glorot_uniform",
            bias_initializer=keras.initializers.RandomNormal(stddev=1e-6),
        )(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer(config: ml_collections.ConfigDict, name: str) -> keras.Model:
    """Transformer block with pre-norm."""
    num_patches = (
        config.num_patches + 1
        if config.classifier == "token"
        else config.num_patches + 0
    )
    encoded_patches = layers.Input((num_patches, config.projection_dim))

    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(encoded_patches)

    # Multi Head Self Attention layer 1.
    attention_output, attention_score = layers.MultiHeadAttention(
        num_heads=config.num_heads,
        key_dim=config.projection_dim,
        dropout=config.dropout_rate,
    )(x1, x1, return_attention_scores=True)
    attention_output = layers.Dropout(config.dropout_rate)(attention_output)

    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(x2)

    # MLP layer 1.
    x4 = mlp(x3, hidden_units=config.mlp_units, dropout_rate=config.dropout_rate)

    # Skip connection 2.
    outputs = layers.Add()([x2, x4])

    return keras.Model(encoded_patches, [outputs, attention_score], name=name)


def transformer_extended(config: ml_collections.ConfigDict, name: str) -> keras.Model:
    """Transformer block with pre-norm. This layer is re-written to port the
    pre-trained JAX weights.
    """
    num_patches = (
        config.num_patches + 1
        if config.classifier == "token"
        else config.num_patches + 0
    )
    encoded_patches = layers.Input((num_patches, config.projection_dim))

    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(encoded_patches)

    # Multi Head Self Attention layer 1.
    attention_output, attention_score = TFViTAttention(config)(
        x1, output_attentions=True
    )

    # Skip connection 1.
    x2 = layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=config.layer_norm_eps)(x2)

    # MLP layer 1.
    x4 = mlp(x3, hidden_units=config.mlp_units, dropout_rate=config.dropout_rate)

    # Skip connection 2.
    outputs = layers.Add()([x2, x4])

    return keras.Model(encoded_patches, [outputs, attention_score], name=name)


class ViTClassifier(keras.Model):
    """Class that collates all the different elements for a Vision Transformer."""

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.projection = keras.Sequential(
            [
                layers.Conv2D(
                    filters=config.projection_dim,
                    kernel_size=(config.patch_size, config.patch_size),
                    strides=(config.patch_size, config.patch_size),
                    padding="VALID",
                    name="conv_projection",
                ),
                layers.Reshape(
                    target_shape=(config.num_patches, config.projection_dim),
                    name="flatten_projection",
                ),
            ],
            name="projection",
        )

        self.positional_embedding = PositionalEmbedding(
            config, name="positional_embedding"
        )
        self.transformer_blocks = [
            transformer(config, name=f"transformer_block_{i}")
            for i in range(config.num_layers)
        ]

        if config.classifier == "token":
            initial_value = tf.zeros((1, 1, config.projection_dim))
            self.cls_token = tf.Variable(
                initial_value=initial_value, trainable=True, name="cls"
            )

        if config.classifier == "gap":
            self.gap_layer = layers.GlobalAvgPool1D()

        self.dropout = layers.Dropout(config.dropout_rate)
        self.layer_norm = layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.classifier_head = layers.Dense(
            config.num_classes,
            kernel_initializer="zeros",
            dtype="float32",
            name="classifier",
        )

    def call(self, inputs, training=True, pre_logits=False):
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.projection(inputs)

        # Append class token if needed.
        if self.config.classifier == "token":
            cls_token = tf.tile(self.cls_token, (n, 1, 1))
            if cls_token.dtype != projected_patches.dtype:
                cls_token = tf.cast(cls_token, projected_patches.dtype)
            projected_patches = tf.concat([cls_token, projected_patches], axis=1)

        # Add positional embeddings to the projected patches.
        encoded_patches = self.positional_embedding(
            projected_patches
        )  # (B, number_patches, projection_dim)
        encoded_patches = self.dropout(encoded_patches)

        if not training:
            attention_scores = dict()

        # Iterate over the number of layers and stack up blocks of
        # Transformer.
        for transformer_module in self.transformer_blocks:
            # Add a Transformer block.
            encoded_patches, attention_score = transformer_module(encoded_patches)
            if not training:
                attention_scores[f"{transformer_module.name}_att"] = attention_score

        # Final layer normalization.
        representation = self.layer_norm(encoded_patches)

        # Pool representation.
        if self.config.classifier == "token":
            encoded_patches = representation[:, 0]
        elif self.config.classifier == "gap":
            encoded_patches = self.gap_layer(representation)

        if pre_logits:
            return encoded_patches

        else:
            # Classification head.
            output = self.classifier_head(encoded_patches)

            if not training:
                return output, attention_scores
            else:
                return output


class ViTClassifierExtended(keras.Model):
    """Class that collates all the different elements for a Vision Transformer.
    This class is for porting the original JAX weights to TF/Keras.
    """

    def __init__(self, config: ml_collections.ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.projection = keras.Sequential(
            [
                layers.Conv2D(
                    filters=config.projection_dim,
                    kernel_size=(config.patch_size, config.patch_size),
                    strides=(config.patch_size, config.patch_size),
                    padding="VALID",
                    name="conv_projection",
                ),
                layers.Reshape(
                    target_shape=(config.num_patches, config.projection_dim),
                    name="flatten_projection",
                ),
            ],
            name="projection",
        )

        init_value = tf.ones(
            (
                1,
                config.num_patches + 1
                if self.config.classifier == "token"
                else config.num_patches,
                config.projection_dim,
            )
        )
        self.positional_embedding = tf.Variable(
            init_value, name="pos_embedding"
        )  # This will be loaded with the pre-trained positional embeddings later.

        self.transformer_blocks = [
            transformer_extended(config, name=f"transformer_block_{i}")
            for i in range(config.num_layers)
        ]  # Extended transformer block to easily load the pre-train variables especially
        # in the attention layers.

        if config.classifier == "token":
            initial_value = tf.zeros((1, 1, config.projection_dim))
            self.cls_token = tf.Variable(
                initial_value=initial_value, trainable=True, name="cls"
            )

        if config.classifier == "gap":
            self.gap_layer = layers.GlobalAvgPool1D()

        self.dropout = layers.Dropout(config.dropout_rate)
        self.layer_norm = layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.classifier_head = layers.Dense(
            config.num_classes,
            kernel_initializer="zeros",
            dtype="float32",
            name="classifier",
        )

    def call(self, inputs, training=True, pre_logits=False):
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.projection(inputs)

        # Append class token if needed.
        if self.config.classifier == "token":
            cls_token = tf.tile(self.cls_token, (n, 1, 1))
            if cls_token.dtype != projected_patches.dtype:
                cls_token = tf.cast(cls_token, projected_patches.dtype)
            projected_patches = tf.concat([cls_token, projected_patches], axis=1)

        # Add positional embeddings to the projected patches.
        encoded_patches = (
            self.positional_embedding + projected_patches
        )  # (B, number_patches, projection_dim)
        encoded_patches = self.dropout(encoded_patches)

        if not training:
            attention_scores = dict()

        # Iterate over the number of layers and stack up blocks of
        # Transformer.
        for transformer_module in self.transformer_blocks:
            # Add a Transformer block.
            encoded_patches, attention_score = transformer_module(encoded_patches)
            if not training:
                attention_scores[f"{transformer_module.name}_att"] = attention_score

        # Final layer normalization.
        representation = self.layer_norm(encoded_patches)

        # Pool representation.
        if self.config.classifier == "token":
            encoded_patches = representation[:, 0]
        elif self.config.classifier == "gap":
            encoded_patches = self.gap_layer(representation)

        if pre_logits:
            return encoded_patches

        else:
            # Classification head.
            output = self.classifier_head(encoded_patches)

            if not training:
                return output, attention_scores
            else:
                return output


def get_augmentation_model(config: ml_collections.ConfigDict, train=True):
    """Augmentation transformation models."""
    if train:
        data_augmentation = keras.Sequential(
            [
                layers.Resizing(config.input_shape[0] + 20, config.input_shape[0] + 20),
                layers.RandomCrop(config.image_size, config.image_size),
                layers.RandomFlip("horizontal"),
                layers.Rescaling(1 / 255.0),
            ],
            name="train_aug",
        )
    else:
        data_augmentation = keras.Sequential(
            [
                layers.Resizing(config.input_shape[0] + 20, config.input_shape[0] + 20),
                layers.CenterCrop(config.image_size, config.image_size),
                layers.Rescaling(1 / 255.0),
            ],
            name="test_aug",
        )
    return data_augmentation
