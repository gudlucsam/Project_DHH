def features_2D_model(mName="mobilenet", input_shape=(224, 224, 3), output_shape=(1024,)):
    """Load Inception CNN Model or MobileNet for Feature Extraction.
    Models are loaded without top layers, suitable for feature extraction.


    Keyword arguments:
    mName -- (str) name of CNN model to use (either mobilenet or inception)
    input_shape -- (str) expected input shape of model ( either (224, 224, 3) or (299, 299, 3))
    output_shape -- (str) expected output shape of model ( either (1024,) or (2048))

    return Keras CNN Model
    """
    
    # load pretrained keras models
    if mName == "mobilenet":
        print("Loading MobileNet for feature  extraction...")

        # load base model with top
        keBaseModel = tf.keras.applications.mobilenet.MobileNet(
            weights="imagenet",
            input_shape = (224, 224, 3),
            include_top = True)

        # We'll extract features at the final pool layer
        keModel = tf.keras.models.Model(
            inputs=keBaseModel.input,
            outputs=keBaseModel.layers[-6].output,
            name=mName + "_without_top_layer_v1") 

    elif mName == "inception":
        print("Loading InceptionV3 for feature  extraction...")

        # load base model with top
        keBaseModel = tf.keras.applications.inception_v3.InceptionV3(
            weights='imagenet',
            include_top=True)

        # We'll extract features at the final pool layer
        keModel = tf.keras.models.Model(
            inputs=keBaseModel.input,
            outputs=keBaseModel.layers[-2].output,
            name=mName + "_without_top_layer_v1") 
    
    else: raise ValueError("Unknown 2D feature extraction model")

    # check input & output dimensions
    tuInputShape = keModel.input_shape[1:]
    tuOutputShape = keModel.output_shape[1:]
    print("Expected input shape %s, output shape %s" % (str(tuInputShape), str(tuOutputShape)))

    if tuInputShape != input_shape:
        raise ValueError("Unexpected input shape")
    if tuOutputShape != output_shape:
        raise ValueError("Unexpected output shape")

    return keModel