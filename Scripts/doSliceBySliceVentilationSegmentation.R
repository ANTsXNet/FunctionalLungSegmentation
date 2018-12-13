library( ANTsR )
library( ANTsRNet )
library( keras )

library( ANTsR )
library( ANTsRNet )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) != 3 )
  {
  helpMessage <- paste0( "Usage:  Rscript doSliceBySliceVentilationSegmentation.R",
    " inputFile inputLungMaskFile outputFilePrefix\n" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  inputMaskFileName <- args[2]
  outputFilePrefix <- args [3]
  }

classes <- c( "background", "Normal", "Defect" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "Ventilation", "ForegroundMask" )
channelSize <- length( imageMods )

resampledSliceSize <- c( 128, 128 )
direction <- 3

unetModel <- createUnetModel2D( c( resampledSliceSize, channelSize ),
  convolutionKernelSize = c( 5, 5 ), deconvolutionKernelSize = c( 5, 5 ),
  numberOfOutputs = numberOfClassificationLabels, dropoutRate = 0.2,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 32 )

cat( "Loading weights file" )
startTime <- Sys.time()
weightsFileName <- "unetModel2DWeights.h5" # getPretrainedNetwork( "lungVentilationSegmentation" )
load_model_weights_hdf5( unetModel, filepath = weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Process input

startTimeTotal <- Sys.time()

cat( "Reading ", inputFileName )
startTime <- Sys.time()
image <- antsImageRead( inputFileName, dimension = 3 )
mask <- antsImageRead( inputMaskFileName, dimension = 3 )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )


cat( "Prediction and decoding (slice-by-slice)." )
startTime <- Sys.time()

imageSize <- dim( image )
numberOfSlices <- imageSize[direction]
originalSliceSize <- imageSize[-direction]

batchX <- array( data = 0, dim = c( numberOfSlices,
  resampledSliceSize, channelSize ) )

for( j in seq_len( numberOfSlices ) )
  {
  imageSlice <- extractSlice( image, j, direction )
  maskSlice <- extractSlice( mask, j, direction )
  if( any( originalSliceSize != resampledSliceSize ) )
    {
    imageSlice <- resampleImage( imageSlice,
      resampledSliceSize, useVoxels = TRUE, interpType = 1 )
    maskSlice <- resampleImage( maskSlice,
      resampledSliceSize, useVoxels = TRUE, interpType = 1 )
    }

  arrayImageSlice <- as.array( imageSlice )
  arrayImageSlice <- ( arrayImageSlice - mean( arrayImageSlice ) ) /
    sd( arrayImageSlice )

  batchX[j,,,1] <- arrayImageSlice
  batchX[j,,,2] <- as.array( maskSlice )
  }

predictedData <- unetModel %>% predict( batchX, verbose = 0 )
probabilitySlices <- decodeUnet( predictedData, imageSlice )

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )


cat( "Writing", outputFilePrefix )
startTime <- Sys.time()
for( i in seq_len( numberOfClassificationLabels ) )
  {
  probabilityArray <- array( data = 0, dim = imageSize )
  for( k in seq_len( numberOfSlices ) )
    {
    probabilitySlice <- probabilitySlices[[k]][[i]]
    if( any( originalSliceSize != resampledSliceSize ) )
      {
      probabilitySlice <- resampleImage( probabilitySlice,
        originalSliceSize, useVoxels = TRUE, interpType = 1 )
      }
    probabilityArray[,,k] <- as.array( probabilitySlice )
    }
  probabilityImage <- as.antsImage( probabilityArray, reference = image )

  antsImageWrite( probabilityImage, paste0( outputFilePrefix, classes[i], ".nii.gz" ) )
  }
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )



