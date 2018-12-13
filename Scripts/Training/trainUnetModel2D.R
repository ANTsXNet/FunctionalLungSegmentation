library( ANTsR )
library( ANTsRNet )
library( keras )

keras::backend()$clear_session()

antsrnetDirectory <- '/Users/ntustison/Pkg/ANTsRNet/'
modelDirectory <- paste0( antsrnetDirectory, 'Models/' )
baseDirectory <- '/Users/ntustison/Data/HeliumLungStudies/DeepVentNet/'

source( paste0( baseDirectory, 'Scripts/unetBatchGenerator2D.R' ) )

classes <- c( "background", "defect/hypo", "normal" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "Ventilation", "ForegroundMask" )
channelSize <- length( imageMods )

dataDirectory <- paste0( baseDirectory, 'Data/' )
trainingImageDirectory <- paste0( dataDirectory, 
  'Ventilation/Training/Images/' )
trainingVentilationFiles <- list.files( path = trainingImageDirectory, 
  pattern = "*N4.nii.gz", full.names = TRUE )

trainingImageFiles <- list()
trainingSegmentationFiles <- list()
trainingTransforms <- list()

for( i in 1:length( trainingVentilationFiles ) )
  {
  subjectId <- basename( trainingVentilationFiles[i] )
  subjectId <- sub( "N4.nii.gz", '', subjectId )

  trainingImageFiles[[i]] <- c(
    trainingVentilationFiles[i],
    paste0( dataDirectory, 'Ventilation/Training/LungMasks/', subjectId, 
      "Mask.nii.gz" )
    )

  trainingSegmentationFiles[[i]] <- paste0( dataDirectory,
    'Ventilation/Training/Segmentations/', subjectId, 
    "Segmentation2Class.nii.gz" )
  if( !file.exists( trainingSegmentationFiles[[i]] ) )
    {
    stop( paste( "Segmentation file", trainingSegmentationFiles[[i]], 
      "does not exist.\n" ) )
    }

  xfrmPrefix <- paste0( dataDirectory, 
    'Ventilation/Training/Template/T_', subjectId, "Mask" )

  fwdtransforms <- c()
  fwdtransforms[1] <- paste0( xfrmPrefix, i-1, 'Warp.nii.gz' )
  fwdtransforms[2] <- paste0( xfrmPrefix, i-1, 'Affine.txt' )
  invtransforms <- c()
  invtransforms[1] <- paste0( xfrmPrefix, i-1, 'Affine.txt' )
  invtransforms[2] <- paste0( xfrmPrefix, i-1, 'InverseWarp.nii.gz' )

  if( !file.exists( fwdtransforms[1] ) || !file.exists( fwdtransforms[2] ) ||
      !file.exists( invtransforms[1] ) || !file.exists( invtransforms[2] ) )
    {
    stop( "Transform ", paste0( xfrmPrefix, i-1 ), " file does not exist.\n" )
    }

  trainingTransforms[[i]] <- list( 
    fwdtransforms = fwdtransforms, invtransforms = invtransforms )
  }

###
#
# Create the Unet model
#
resampledImageSize <- c( 128, 128 )

direction <- 3

unetModel <- createUnetModel2D( c( resampledImageSize, channelSize ), 
  convolutionKernelSize = c( 5, 5 ), deconvolutionKernelSize = c( 5, 5 ),
  numberOfClassificationLabels = numberOfClassificationLabels, dropoutRate = 0.2,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 32 )

# load_model_weights_hdf5( unetModel, 
#   filepath = paste0( dataDirectory, 'Ventilation/Models/unetModel2DWeights.h5' ) )

unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
  optimizer = optimizer_adam( lr = 0.00001 ),  
  metrics = c( multilabel_dice_coefficient ) )

###
#
# Set up the training generator
#

batchSize <- 32L

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfTrainingData <- length( trainingImageFiles )
sampleIndices <- sample( numberOfTrainingData )

validationSplit <- floor( 0.8 * length( numberOfTrainingData ) )
trainingIndices <- sampleIndices[1:validationSplit]
validationIndices <- sampleIndices[( validationSplit + 1 ):batchSize]

trainingData <- unetImageBatchGenerator2D$new( 
  imageList = trainingImageFiles[trainingIndices], 
  segmentationList = trainingSegmentationFiles[trainingIndices], 
  transformList = trainingTransforms[trainingIndices], 
  referenceImageList = trainingImageFiles, 
  referenceTransformList = trainingTransforms )

trainingDataGenerator <- trainingData$generate( batchSize = batchSize,
  direction = direction, sliceSamplingRate = 0.5,
  resampledSliceSize = resampledImageSize )

validationData <- unetImageBatchGenerator2D$new( 
  imageList = trainingImageFiles[validationIndices], 
  segmentationList = trainingSegmentationFiles[validationIndices], 
  transformList = trainingTransforms[validationIndices], 
  referenceImageList = trainingImageFiles, 
  referenceTransformList = trainingTransforms )

validationDataGenerator <- trainingData$generate( batchSize = batchSize,
  direction = direction, sliceSamplingRate = 0.5,
  resampledSliceSize = resampledImageSize )

###
#
# Run training
#

track <- unetModel$fit_generator( 
  generator = reticulate::py_iterator( trainingDataGenerator ), 
  steps_per_epoch = ceiling( 0.25 * 0.8 * 0.5 * 128 * numberOfTrainingData  / batchSize ),
  epochs = 200,
  validation_data = reticulate::py_iterator( validationDataGenerator ),
  validation_steps = ceiling( 0.25 * 0.2 * 0.5 * 128 * numberOfTrainingData  / batchSize ),
  callbacks = list( 
    callback_model_checkpoint( paste0( dataDirectory, "Ventilation/unetModel2DWeights.h5" ), 
      monitor = 'loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'loss', factor = 0.1,
       verbose = 1, patience = 10, mode = 'auto' )
      # ,
    #  callback_early_stopping( monitor = 'val_loss', min_delta = 0.001, 
    #    patience = 10 ),
  )
)  







