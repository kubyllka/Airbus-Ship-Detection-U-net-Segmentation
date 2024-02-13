import tensorflow as tf
import keras
from keras.models import load_model
import pandas as pd
import numpy as np
import cv2
import argparse

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
        [pixel.append( rle[i] - 1 ) if i % 2 == 0 else pixel_count.append( rle[i] ) for i in range( 0, len( rle ) )]
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


def encode_rle(mask_img):
    """
    Function to encode a binary mask image using RLE (Run-Length Encoding).

    Parameters:
        mask_img (numpy.ndarray): The binary mask image to be encoded.

    Returns:
        str: The RLE encoded pixel values.
    """
    # Flatten the mask image and convert it to a list of integers
    mask_flat = mask_img.T.flatten()

    # Initialize variables
    rle = []
    count = 0
    current_pixel = -1

    # Iterate through the flattened mask image
    for i, pixel in enumerate( mask_flat ):
        if pixel == 1.:  # Check if the pixel value is 1. (indicating object presence)
            if current_pixel == -1:
                current_pixel = i
                count = 1
            else:
                count += 1
        else:
            if count > 0:
                rle.extend( [current_pixel, count] )
            current_pixel = -1
            count = 0

    # Append the last count and pixel value to the RLE list if count is non-zero
    if count > 0:
        rle.extend( [current_pixel, count] )

    if len( rle ):
        encoded_rle = ' '.join( map( str, rle ) )
    else:
        encoded_rle = pd.NA

    return encoded_rle


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


def result_prediction(df, model, path_to_folder):
    """
    Function to generate predictions for images in a DataFrame and update the DataFrame with the encoded pixel masks.

    Args:
        df (pandas.DataFrame): DataFrame containing image IDs and associated information.
        model (keras.Model): Model used for predictions.
        path_to_folder (str): Path to the folder containing the images.

    Returns:
        pandas.DataFrame: DataFrame with updated predictions.
    """
    # Create a copy of the original DataFrame to store the updated predictions
    updated_df = df.copy()

    # Iterate over each row in the DataFrame `df`
    for i in range( 0, len(df)):
        # Get the path to the image from the 'ImageId' column in DataFrame `df`
        image_path = path_to_folder + df['ImageId'].iloc[i]
        # Read the image at the specified path and convert it to RGB format
        image = cv2.imread( image_path )
        image = cv2.cvtColor( image, cv2.COLOR_BGR2RGB )
        # Add a dimension to create a batch of images
        image = np.expand_dims( image, axis=0 )
        # Pass the image to the model and get the predicted masks
        predictions = model.predict( np.array( image ) )
        # Determine the threshold for binarizing the mask
        threshold = np.mean( predictions.squeeze() )
        # Apply the threshold to the predicted values to obtain a binary mask
        mask = np.where( predictions.squeeze() > threshold, 1, 0 )
        # Encode the mask into an RLE vector
        encoded_pixels = encode_rle( mask )
        # Write the encoded pixels to the corresponding 'EncodedPixels' column in the updated DataFrame
        updated_df.at[i, 'EncodedPixels'] = encoded_pixels

    return updated_df

def main():
    parser = argparse.ArgumentParser(description='Ship detection')
    parser.add_argument('input_file_path', type=str, help='Path to the input CSV file (e.g., test_reviews.csv)')
    parser.add_argument('input_img_path', type=str, help='Path to the input images directory')
    parser.add_argument('--output_file_path', type=str, default='result.csv', help='Path to the output file')
    parser.add_argument('model_path', type=str, help='Path to the model')
    args = parser.parse_args()

    model_path = args.model_path
    test_dir_csv = args.input_file_path
    test_dir_img = args.input_img_path
    output_file_path = args.output_file_path

    keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient
    keras.utils.get_custom_objects()['dice_loss'] = dice_loss

    # Loading the model
    model = load_model(model_path, custom_objects={'BatchNormalization': keras.layers.BatchNormalization})

    test_data = pd.read_csv(test_dir_csv)
    updated_df = result_prediction(test_data, model, test_dir_img)
    updated_df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main()




