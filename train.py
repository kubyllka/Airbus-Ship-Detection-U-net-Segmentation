import numpy as np
import pandas as pd
import cv2
import imageio
import tensorflow as tf
import tensorflow.python.keras
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import imgaug.augmenters as iaa
from sklearn.utils import shuffle
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import argparse
import matplotlib.pyplot as plt



def decode_rle(encoded_pixels, shape):
    """
    Function to decode the RLE (Run-Length Encoding) encoded pixels into a binary mask image.

    Parameters:
        encoded_pixels (str): The RLE encoded pixel values.
        shape (tuple): The shape of the mask image (height, width).

    Returns:
        numpy.ndarray: The decoded binary mask image.

    """
    # Create an array to store the mask image
    mask_img = np.zeros( (shape[0] * shape[1], 1), dtype=np.float32 )

    # Check if the RLE encoding is not NaN (not a null value)
    if pd.notna( encoded_pixels ):
        # Split the encoded pixels into a list of integers
        rle = list( map( int, encoded_pixels.split( ' ' ) ) )
        pixel, pixel_count = [], []
        # Separate the pixel values and their counts into separate lists
        [pixel.append(rle[i] - 1) if i % 2 == 0 else pixel_count.append( rle[i] ) for i in range( 0, len( rle ) )]
        # Create a list of pixel ranges based on the pixel values and counts
        rle_pixels = [list( range( pixel[i], pixel[i] + pixel_count[i] ) ) for i in range( 0, len( pixel ) )]
        # Flatten the list of pixel ranges into a single list of pixel indices
        rle_mask_pixels = sum( rle_pixels, [] )

        # Try to set the corresponding pixels in the mask image to 1 based on the RLE indices
        try:
            mask_img[rle_mask_pixels] = 1.
        # Catch any potential IndexError (e.g., if the RLE indices exceed the mask image dimensions)
        except IndexError:
            pass
    # Reshape the flattened mask image array into the original shape (transposed)
    return np.reshape( mask_img, shape ).T

def combine_masks(encoded_pixels):
    """
    Function to combine multiple RLE-encoded masks into a single string.

    Parameters:
        encoded_pixels (list): A list of RLE-encoded pixel values.

    Returns:
        str: A string containing the combined RLE-encoded masks.

    """
    masks = ' '.join( map( str, encoded_pixels ) )
    return masks


def process_image(image_id, group):
    """
    Function to process an image group by combining multiple masks into a single combined mask.

    Parameters:
        image_id (str): The ID of the image.
        group (pandas.DataFrame): A DataFrame containing the image group.

    Returns:
        tuple: A tuple containing the image ID, combined mask, and the number of masks in the group.

    """
    # Extract the list of encoded pixels from the DataFrame group
    encoded_pixels = group['EncodedPixels'].tolist()

    # Check if all encoded pixels are not null
    if np.all( pd.notna( encoded_pixels ) ):
        # Combine the masks into a single string
        combined_mask = combine_masks( encoded_pixels )
        # Return the image ID, combined mask, and number of masks
        return image_id, combined_mask, len( group )
    else:
        # If any encoded pixels are null, return None values
        return image_id, None, 0


def conv_block(inputs, num_filters):
    """
    Convolutional block consisting of two convolutional layers with batch normalization and ReLU activation.

    Args:
        inputs (tensorflow.Tensor): Input tensor.
        num_filters (int): Number of filters for convolutional layers.

    Returns:
        tensorflow.Tensor: Output tensor.
    """
    x = Conv2D( num_filters, 3, padding="same" )( inputs )
    x = BatchNormalization()( x )
    x = Activation( "relu" )( x )

    x = Conv2D( num_filters, 3, padding="same" )( x )
    x = BatchNormalization()( x )
    x = Activation( "relu" )( x )

    return x


def encoder_block(inputs, num_filters):
    """
    Encoder block comprising a convolutional block followed by max pooling.

    Args:
        inputs (tensorflow.Tensor): Input tensor.
        num_filters (int): Number of filters for the convolutional block.

    Returns:
        Tuple[tensorflow.Tensor, tensorflow.Tensor]: Output tensors from the convolutional block and max pooling.
    """
    x = conv_block( inputs, num_filters )
    p = MaxPool2D( (2, 2) )( x )
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    """
    Decoder block consisting of transposed convolution, concatenation with skip connections, and convolutional block.

    Args:
        inputs (tensorflow.Tensor): Input tensor.
        skip_features (tensorflow.Tensor): Skip connection tensor from encoder block.
        num_filters (int): Number of filters for the convolutional block.

    Returns:
        tensorflow.Tensor: Output tensor.
    """
    x = Conv2DTranspose( num_filters, 2, strides=2, padding="same" )( inputs )
    x = Concatenate()( [x, skip_features] )
    x = conv_block( x, num_filters )
    return x


def build_unet(input_shape):
    """
    Function to build the U-Net model architecture.

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels).

    Returns:
        tensorflow.keras.Model: U-Net model.
    """
    inputs = Input( input_shape )

    s1, p1 = encoder_block( inputs, 32 )
    s2, p2 = encoder_block( p1, 64 )
    s3, p3 = encoder_block( p2, 128 )
    s4, p4 = encoder_block( p3, 256 )

    b1 = conv_block( p4, 512 )

    d1 = decoder_block( b1, s4, 256 )
    d2 = decoder_block( d1, s3, 128 )
    d3 = decoder_block( d2, s2, 64 )
    d4 = decoder_block( d3, s1, 32 )

    outputs = Conv2D( 1, 1, padding="same", activation="sigmoid" )( d4 )

    model = Model( inputs, outputs, name="UNET" )
    return model


def dice_coefficient(y_true, y_pred):
    """
    Calculate the Dice coefficient, a metric used for evaluating segmentation performance.

    Args:
        y_true (tensorflow.Tensor): True binary labels.
        y_pred (tensorflow.Tensor): Predicted binary labels.

    Returns:
        float: Dice coefficient value.
    """
    smooth = 1e-15

    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):
    """
    Calculate the Dice loss, which is 1 minus the Dice coefficient.

    Args:
        y_true (tensorflow.Tensor): True binary labels.
        y_pred (tensorflow.Tensor): Predicted binary labels.

    Returns:
        float: Dice loss value.
    """
    return 1. - dice_coefficient( y_true, y_pred )


def preprocess_image_mask(image, mask):
    """
    Preprocesses the input image and mask by applying data augmentation and normalization.

    Parameters:
        images (numpy.ndarray): The batch of input images.
        mask (numpy.ndarray): The batch of corresponding segmentation masks.

    Returns:
        tuple: A tuple containing the preprocessed images and masks.
    """

    # Data augmentation
    seq = iaa.Sequential( [
        iaa.Fliplr( 0.5 ),  # Apply horizontal flips with a probability of 0.5
        iaa.Flipud( 0.5 ),  # Apply vertical flips with a probability of 0.5
        iaa.Affine( rotate=(-10, 10) ),  # Apply random rotations between -10 and 10 degrees
        iaa.GaussianBlur( sigma=(0, 0.3) ),  # Apply Gaussian blur with a sigma between 0 and 0.5
    ] )

    mask = mask.astype( np.uint8 )  # Convert to uint8 data type
    mask = np.expand_dims( mask, axis=-1 )  # Add an extra dimension for C
    segmap = SegmentationMapsOnImage( mask, shape=image.shape )

    # Apply augmentation to the images and masks
    augmented = seq( image=image, segmentation_maps=segmap )
    images_augmented = augmented[0]  # Access the augmented images
    mask_augmented = augmented[1].get_arr()  # Access the augmented segmentation masks

    # print(mask_augmented)

    # Normalize pixel values to the range [0, 1]
    images_augmented = images_augmented / 255.0

    return images_augmented, mask_augmented


def data_generator(X, y, batch_size, path, shape):
    """
    Generator function to yield batches of images and masks for training.

    Parameters:
        X (list): List of image filenames.
        y (list): List of encoded mask values.
        batch_size (int): Size of each batch.

    Yields:
        tuple: A tuple containing the batch of images and masks.
    """
    while True:
        # Iterate over the entire dataset in batches
        for i in range( 0, len( X ), batch_size ):
            batch_X = []
            batch_y = []
            # Iterate over the current batch
            for j in range( i, min( i + batch_size, len( X ) ) ):
                # Load and preprocess the image
                image_path = path + X[j]
                image = cv2.imread( image_path )
                image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )  # Convert image to RGB format
                # Decode the encoded mask and preprocess it
                mask = decode_rle( y[j], shape)
                image, mask = preprocess_image_mask( image, mask )
                batch_X.append( image )
                batch_y.append( mask )

            # Yield the batch of images and masks
            yield np.array( batch_X ), np.array( batch_y )

def plot_train_history(history):
    """
    Function to plot the training history including loss and dice score.

    Args:
        history: History object returned by the `fit` method of a Keras model.
    """
    # Get training history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_dice_score = history.history['dice_coefficient']
    val_dice_score = history.history['val_dice_coefficient']

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation dice score
    plt.figure(figsize=(10, 5))
    plt.plot(train_dice_score, label='Training Dice Score', color='blue')
    plt.plot(val_dice_score, label='Validation Dice Score', color='orange')
    plt.title('Training and Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.show()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser( description='Training script' )
    parser.add_argument( 'input_file_path', type=str, help='Path to the input CSV file (e.g., test_reviews.csv)' )
    parser.add_argument( 'input_img_path', type=str, help='Path to the input images directory' )
    parser.add_argument( '--epochs', type=int, default=2, help='Number of epochs for training' )
    parser.add_argument( '--batch_size', type=int, default=2, help='Batch size for training' )
    parser.add_argument( '--learning_rate', type=float, default=0.01, help='Learning rate for training' )
    parser.add_argument( '--random_seed', type=int, default=4747, help='Random seed' )
    parser.add_argument( '--number_img_to_train', type=int, default=100,
                         help='Number images for training and validating (Random choice from data)' )
    parser.add_argument( '--percentage_for_val_split', type=float, default=0.15,
                         help='Percentage of images for val split (from images to train)' )
    parser.add_argument( '--percentage_img_positive', type=float, default=0.7,
                         help='Percentage of images that have a positive class' )
    args = parser.parse_args()

    # Extract command-line arguments
    train_dir_csv = args.input_file_path
    train_dir_img = args.input_img_path
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    random_state_num = args.random_seed
    num_images_train_val = args.number_img_to_train
    percentage_val_split = args.percentage_for_val_split
    percentage_img_with_ships = args.percentage_img_positive

    # Read CSV file into DataFrame
    df = pd.read_csv( train_dir_csv )
    #print( df.head() )

    # Extract image dimensions from the first image
    image_file = df['ImageId'].iloc[0]
    image_path = train_dir_img + image_file
    image = cv2.imread( image_path )
    height = image.shape[0]
    width = image.shape[1]
    num_channels = image.shape[2]

    # Process image data
    grouped_data = df.groupby( 'ImageId' )
    image_data = [(group_name, group) for group_name, group in grouped_data]
    processed_results = []
    for group_name, group in image_data:
        result = process_image( group_name, group )
        processed_results.append( result )
    result_df = pd.DataFrame( processed_results, columns=['ImageId', 'CombinedMask', 'NumShips'] )
    # print( result_df.head() )

    # Split data into images with ships and without ships
    has_ships = result_df[result_df['NumShips'] != 0]
    no_ships = result_df[result_df['NumShips'] == 0]
    number_img_train_ships = int( percentage_img_with_ships * num_images_train_val )
    number_img_train_noships = num_images_train_val - number_img_train_ships
    has_ships = shuffle( has_ships, random_state=random_state_num )
    no_ships = shuffle( no_ships, random_state=random_state_num )
    has_ships = has_ships[: number_img_train_ships]
    no_ships = no_ships[: number_img_train_noships]

    # Split data into training and validation sets
    X_train_pos, X_val_pos, y_train_pos, y_val_pos = train_test_split( has_ships['ImageId'], has_ships['CombinedMask'],
                                                                       test_size=percentage_val_split,
                                                                       random_state=random_state_num )
    X_train_neg, X_val_neg, y_train_neg, y_val_neg = train_test_split( no_ships['ImageId'], no_ships['CombinedMask'],
                                                                       test_size=percentage_val_split,
                                                                       random_state=random_state_num )
    X_train = np.concatenate( (X_train_pos, X_train_neg) )
    X_val = np.concatenate( (X_val_pos, X_val_neg) )
    y_train = np.concatenate( (y_train_pos, y_train_neg) )
    y_val = np.concatenate( (y_val_pos, y_val_neg) )
    X_train, y_train = shuffle( X_train, y_train, random_state=random_state_num )
    X_val, y_val = shuffle( X_val, y_val, random_state=random_state_num )

    # Build and compile the U-Net model
    model = build_unet( (height, width, num_channels) )
    model.compile( optimizer=keras.optimizers.Adam( learning_rate=lr ),
                   loss=[dice_loss],
                   metrics=[dice_coefficient] )
    checkpoint = ModelCheckpoint( "model.keras", monitor='val_loss', verbose=1, save_best_only=True )
    early_stopping = EarlyStopping( monitor='val_loss', patience=5, restore_best_weights=True )

    # Define training parameters
    steps_per_epoch = len( X_train ) // batch_size
    validation_steps = len( X_val ) // batch_size

    # Train the model
    history = model.fit( data_generator( X_train, y_train, batch_size, train_dir_img, (height, width) ),
                         steps_per_epoch=steps_per_epoch,
                         epochs=num_epochs,
                         validation_data=data_generator( X_val, y_val, batch_size, train_dir_img, (height, width) ),
                         validation_steps=validation_steps,
                         callbacks=[checkpoint, early_stopping] )

    # Evaluate the model
    loss, dice_score = model.evaluate( data_generator( X_val, y_val, batch_size, train_dir_img, (height, width) ),
                                       steps=validation_steps )

    # Plot the training history
    plot_train_history( history )


if __name__ == "__main__":
    main()
