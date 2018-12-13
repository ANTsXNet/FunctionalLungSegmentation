#' @export

unetImageBatchGenerator2D <- R6::R6Class( "UnetImageBatchGenerator2D",

  public = list( 
    
    imageList = NULL,

    segmentationList = NULL,

    transformList = NULL,

    referenceImageList = NULL,

    referenceTransformList = NULL,

    pairwiseIndices = NULL,

    initialize = function( imageList = NULL, segmentationList = NULL, 
      transformList = NULL, referenceImageList = NULL, 
      referenceTransformList = NULL )
      {
        
      if( !usePkg( "ANTsR" ) )
        {
        stop( "Please install the ANTsR package." )
        }

      if( !is.null( imageList ) )
        {
        self$imageList <- imageList
        } else {
        stop( "Input images must be specified." )
        }

      if( !is.null( segmentationList ) )
        {
        self$segmentationList <- segmentationList
        } else {
        stop( "Input segmentation images must be specified." )
        }

      if( !is.null( transformList ) )
        {
        self$transformList <- transformList
        } else {
        stop( "Input transforms must be specified." )
        }

      if( is.null( referenceImageList ) || 
        is.null( referenceTransformList ) )
        {
        self$referenceImageList <- imageList
        self$referenceTransformList <- transformList
        } else {
        self$referenceImageList <- referenceImageList
        self$referenceTransformList <- referenceTransformList
        }

      self$pairwiseIndices <- expand.grid( source = 1:length( self$imageList ), 
        reference = 1:length( self$referenceImageList ) )  

      # shuffle the pairs
      self$pairwiseIndices <- 
        self$pairwiseIndices[sample.int( nrow( self$pairwiseIndices ) ),]
      },

    generate = function( batchSize = 32L, resampledSliceSize = c( 128, 128 ), 
      direction = 2, sliceSamplingRate = 0.2, doRandomHistogramMatching = TRUE )
      {
      # shuffle the source data
      sampleIndices <- sample( length( self$imageList ) )
      self$imageList <- self$imageList[sampleIndices]
      self$segmentationList <- self$segmentationList[sampleIndices]
      self$transformList <- self$transformList[sampleIndices]

      # shuffle the reference data
      sampleIndices <- sample( length( self$referenceImageList ) )
      self$referenceImageList <- self$referenceImageList[sampleIndices]
      self$referenceTransformList <- self$referenceTransformList[sampleIndices]

      currentPassCount <- 1L

      function() 
        {
        # Shuffle the data after each complete pass 

        if( currentPassCount >= nrow( self$pairwiseIndices ) )
          {
          # shuffle the source data
          sampleIndices <- sample( length( self$imageList ) )
          self$imageList <- self$imageList[sampleIndices]
          self$segmentationList <- self$segmentationList[sampleIndices]
          self$transformList <- self$transformList[sampleIndices]

          # shuffle the reference data
          sampleIndices <- sample( length( self$referenceImageList ) )
          self$referenceImageList <- self$referenceImageList[sampleIndices]
          self$referenceTransformList <- self$referenceTransformList[sampleIndices]

          currentPassCount <- 1L
          }

        rowIndices <- currentPassCount + 0:( batchSize - 1L )

        outOfBoundsIndices <- which( rowIndices > nrow( self$pairwiseIndices ) )
        while( length( outOfBoundsIndices ) > 0 )
          {
          rowIndices[outOfBoundsIndices] <- rowIndices[outOfBoundsIndices] - 
            nrow( self$pairwiseIndices )
          outOfBoundsIndices <- which( rowIndices > nrow( self$pairwiseIndices ) )
          }
        batchIndices <- self$pairwiseIndices[rowIndices,]

        batchImages <- self$imageList[batchIndices$source]
        batchSegmentations <- self$segmentationList[batchIndices$source]
        batchTransforms <- self$transformList[batchIndices$source]

        batchReferenceImages <- self$referenceImageList[batchIndices$reference]
        batchReferenceTransforms <- self$referenceTransformList[batchIndices$reference]

        channelSize <- length( batchImages[[1]] )

        batchX <- array( data = 0, dim = c( batchSize, resampledSliceSize, channelSize ) )
        batchY <- array( data = 0, dim = c( batchSize, resampledSliceSize ) )

        currentPassCount <<- currentPassCount + batchSize

        i <- 1
        while( i <= batchSize )
          {
          subjectBatchImages <- batchImages[[i]]  

          referenceX <- antsImageRead( batchReferenceImages[[i]][1], dimension = 3 )
          referenceXfrm <- batchReferenceTransforms[[i]]
          imageSize <- dim( referenceX )
         
          sourceXfrm <- batchTransforms[[i]]

          boolInvert <- c( TRUE, FALSE, FALSE, FALSE )
          transforms <- c( referenceXfrm$invtransforms[1], 
            referenceXfrm$invtransforms[2], sourceXfrm$fwdtransforms[1],
            sourceXfrm$fwdtransforms[2] )

          numberOfSlices <- imageSize[direction]
          numberOfExtractedSlices <- round( sliceSamplingRate * numberOfSlices )
          slicesToExtract <- sample.int( n = imageSize[direction], 
            size = numberOfExtractedSlices )

          sourceY <- antsImageRead( batchSegmentations[[i]], dimension = 3 )

          warpedImageY <- antsApplyTransforms( referenceX, sourceY, 
            interpolator = "genericLabel", transformlist = transforms,
            whichtoinvert = boolInvert  )

          # Randomly "flip a coin" to see if we perform histogram matching.

          doPerformHistogramMatching <- FALSE
          if( doRandomHistogramMatching == TRUE )
            {
            doPerformHistogramMatching <- sample( c( TRUE, FALSE ), size = 1 )
            }

          warpedImagesX <- list()
          for( j in seq_len( channelSize ) )
            {  
            sourceX <- antsImageRead( subjectBatchImages[j], dimension = 3 )

            warpedImagesX[[j]] <- antsApplyTransforms( referenceX, sourceX, 
              interpolator = "linear", transformlist = transforms,
              whichtoinvert = boolInvert )

            if( doPerformHistogramMatching )
              {
              warpedImagesX[[j]] <- histogramMatchImage( warpedImagesX[[j]],                 
                antsImageRead( batchReferenceImages[[i]][j], dimension = 3 ),
                numberOfHistogramBins = 64, numberOfMatchPoints = 16 )
              }

            # Truncate and rescale image intensity

            warpedArray <- as.array( warpedImagesX[[j]] )

            # truncateQuantiles <- quantile( as.vector( warpedArray ), 
            #   probs = c( 0.01, 0.99 ) )
            # warpedArray[ which( warpedArray < truncateQuantiles[1] )] <- 
            #   truncateQuantiles[1]
            # warpedArray[ which( warpedArray > truncateQuantiles[2] )] <- 
            #   truncateQuantiles[2]
             
            # warpedArray <- ( warpedArray - min( warpedArray ) ) / 
            #   ( max( warpedArray ) - min( warpedArray ) )
            warpedArray <- ( warpedArray - mean( warpedArray ) ) / sd( warpedArray )

            warpedImagesX[[j]] <- as.antsImage( warpedArray, 
              reference = warpedImagesX[[j]] )
            }

          for( k in seq_len( numberOfExtractedSlices ) )
            {
            sliceWarpedImageY <- extractSlice( warpedImageY, 
              slicesToExtract[k], direction )
            
            if( any( dim( sliceWarpedImageY ) != resampledSliceSize ) )
              {
              sliceWarpedArrayY <- as.array( resampleImage( sliceWarpedImageY, 
                resampledSliceSize, useVoxels = TRUE, interpType = 1 ) )
              } else {
              sliceWarpedArrayY <- as.array( sliceWarpedImageY )
              }

            if( sum( sliceWarpedArrayY ) == 0 )
              {
              next
              }
            # antsImageWrite( as.antsImage( sliceWarpedArrayY ), "~/Desktop/arrayY.nii.gz" )
            batchY[i,,] <- sliceWarpedArrayY

            for( j in seq_len( channelSize ) )
              {  
              sliceWarpedImageX <- extractSlice( warpedImagesX[[j]], 
                slicesToExtract[k], direction )

              if( any( dim( sliceWarpedImageY ) != resampledSliceSize ) )
                {
                sliceWarpedArrayX <- as.array( resampleImage( sliceWarpedImageX, 
                  resampledSliceSize, useVoxels = TRUE, interpType = 0 ) )
                } else {
                sliceWarpedArrayX <- as.array( sliceWarpedImageX )
                }  

              # antsImageWrite( as.antsImage( sliceWarpedArrayX ), "~/Desktop/arrayX.nii.gz" )
              # readline( prompt = "Press [enter] to continue\n" )
              batchX[i,,,j] <- sliceWarpedArrayX
              }

            i <- i + 1
            if( i > batchSize )
              {
              break  
              }
            }
          }
  
        segmentationLabels <- sort( unique( as.vector( batchY ) ) )

        encodedBatchY <- encodeUnet( batchY, segmentationLabels )  

        return( list( batchX, encodedBatchY ) )        
        }   
      }
    )
  )