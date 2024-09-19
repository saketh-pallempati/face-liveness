import {
  FaceMatcher,
  LabeledFaceDescriptors,
  detectAllFaces,
  detectSingleFace,
  draw,
  loadFaceLandmarkModel,
  loadFaceRecognitionModel,
  loadSsdMobilenetv1Model,
  matchDimensions,
  resizeResults,
  TinyFaceDetectorOptions,
} from 'face-api.js';
import { useCallback, useEffect, useRef, useState } from 'react';

const MODEL_URL = '/models';
const FACE_MATCHER_THRESHOLD = 0.6;
const VALID_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

import styles from './faceRecognition.module.scss';

export default function FaceRecognition() {
  const [dataSetImages, setDataSetImages] = useState<Array<any>>([]);
  const [faceMatcher, setFaceMatcher] = useState<FaceMatcher | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loaderMsg, setLoaderMsg] = useState('');
  const [queryImage, setQueryImage] = useState<string | null>(null);
  const [recognitionError, setRecognitionError] = useState('');
  const [livenessDetected, setLivenessDetected] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const queryCanvasElement = useRef<HTMLCanvasElement | null>(null);
  const queryImageElement = useRef<HTMLImageElement | null>(null);
  const refImgElements = useRef<(HTMLImageElement | null)[]>([]);

  const addImageRef = (index: number, ref: HTMLImageElement | null) => {
    refImgElements.current[index] = ref;
  };

  const processImagesForRecognition = useCallback(async () => {
    if (!dataSetImages) return;
    setIsLoading(true);
    setLoaderMsg('Please wait while images are being processed...');
    let labeledFaceDescriptors = [];
    labeledFaceDescriptors = await Promise.all(
      refImgElements.current?.map(async (imageEle) => {
        if (imageEle) {
          const label = imageEle?.alt.split(' ')[0];
          const faceDescription = await detectSingleFace(imageEle)
            .withFaceLandmarks()
            .withFaceDescriptor();
          if (!faceDescription) {
            throw new Error(`no faces detected for ${label}`);
          }

          const faceDescriptors = [faceDescription.descriptor];
          return new LabeledFaceDescriptors(label, faceDescriptors);
        }
      })
    );

    const faceMatcher = new FaceMatcher(
      labeledFaceDescriptors,
      FACE_MATCHER_THRESHOLD
    );

    setFaceMatcher(faceMatcher);
    setIsLoading(false);
  }, [dataSetImages]);

  const detectLiveness = async () => {
    if (videoRef.current) {
      const detections = await detectAllFaces(videoRef.current, new TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceExpressions();

      if (detections.length > 0) {
        const expressions = detections[0].expressions;
        const isBlinking = expressions.neutral > 0.5 && (expressions.happy > 0.1 || expressions.surprised > 0.1);
        setLivenessDetected(isBlinking);
      }
    }
  };

  const startVideo = () => {
    navigator.mediaDevices.getUserMedia({ video: {} })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(err => console.error('Error accessing webcam: ', err));
  };

  const loadRecognizedFaces = async () => {
    if (dataSetImages.length === 0) {
      setRecognitionError('No data set found for recognition.');
    } else if (!queryImageElement.current) {
      setRecognitionError('Please upload query image for recognition.');
    }
    if (
      queryCanvasElement.current &&
      queryImageElement.current &&
      faceMatcher
    ) {
      setRecognitionError('');
      const resultsQuery = await detectAllFaces(queryImageElement.current)
        .withFaceLandmarks()
        .withFaceDescriptors();

      await matchDimensions(
        queryCanvasElement.current,
        queryImageElement.current
      );

      const results = await resizeResults(resultsQuery, {
        width: (queryImageElement.current as HTMLImageElement).width,
        height: (queryImageElement.current as HTMLImageElement).height,
      });

      const queryDrawBoxes = results.map((res) => {
        const bestMatch = faceMatcher.findBestMatch(res.descriptor);
        return new draw.DrawBox(res.detection.box, {
          label: bestMatch.toString(),
        });
      });

      queryDrawBoxes.forEach((drawBox) =>
        drawBox.draw(
          queryCanvasElement.current as unknown as HTMLCanvasElement
        )
      );

      // Detect liveness
      await detectLiveness();
    }
    setIsLoading(false);
  };

  const setImagesForRecognition = (event: any) => {
    const files = Array.from(event.target.files || []);

    // Limit the number of selected images
    if (files.length > 20) {
      setRecognitionError('You can select a maximum of 20 images.');
      return;
    }

    if (files) {
      const images = [];
      refImgElements.current = [];
      setRecognitionError('');
      for (let index = 0; index < event.target.files.length; index++) {
        const image = event.target.files[index];
        if (VALID_IMAGE_TYPES.includes(image.type)) {
          images.push({
            name: image.name,
            src: URL.createObjectURL(image),
          });
        }
      }
      setDataSetImages(images);
    }
  };

  const handleQueryImage = (event: any) => {
    if (event?.target?.files && event?.target?.files[0]) {
      setRecognitionError('');
      const image = event.target.files[0];
      setQueryImage(URL.createObjectURL(image));
      if (queryCanvasElement.current) {
        const canvasEle = (
          queryCanvasElement.current as any
        ).getContext('2d');
        canvasEle?.reset();
      }
    }
  };

  const loadModels = async () => {
    setLoaderMsg('Please wait while SSD Mobile net model is loading...');
    await loadSsdMobilenetv1Model(MODEL_URL);
    setLoaderMsg('Please wait while face landmark model is loading...');
    await loadFaceLandmarkModel(MODEL_URL);
    setLoaderMsg('Please wait while face expression model is loading...');
    await loadFaceRecognitionModel(MODEL_URL);
    setIsLoading(false);
  };

  useEffect(() => {
    loadModels();
    startVideo();
  }, []);

  useEffect(() => {
    if (dataSetImages?.length > 0) {
      processImagesForRecognition();
    }
  }, [dataSetImages, processImagesForRecognition]);

  useEffect(() => {
    const interval = setInterval(() => {
      detectLiveness();
    }, 1000); // Check for liveness every second
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      {isLoading && <div>{loaderMsg}</div>}
      <div className={`container ${styles.container}`}>
        <div
          className={`${styles.imageSection} ${styles.multiImageSection}`}>
          <div className={styles.twoSectionPreview}>
            <div className={styles.dataSetSection}>
              <h4>Original Data</h4>
              <div>
                {dataSetImages?.map((image, index) => {
                  return (
                    <div
                      className={styles.imageArea}
                      key={`data-set-${index}`}>
                      <img
                        ref={(imageRef) =>
                          addImageRef(index, imageRef)
                        }
                        src={image.src}
                        alt={image.name}
                        width={100}
                        height={100}
                      />
                      <span>{image.name}</span>
                    </div>
                  );
                })}
              </div>
              <label
                htmlFor='multiFileSelect'
                className={styles.fileUpload}>
                <span>
                  <i className='bi bi-upload'></i>
                </span>
                Upload image data set for face recognition
              </label>
              <input
                id='multiFileSelect'
                type='file'
                onChange={setImagesForRecognition}
                multiple
                accept='image/jpeg, image/png, image/webp'
                hidden
              />
            </div>
            <div className={styles.queryImageSection}>
              <h4>Query Image</h4>
              <div>
                {queryImage && (
                  <>
                    <img
                      ref={queryImageElement}
                      src={queryImage}
                      alt='Image to be recognized for face, Face Recognition'
                      width={500}
                      height={500}
                    />
                    <canvas
                      ref={queryCanvasElement}
                      className={styles.canvas}
                      width={500}
                      height={500}
                    />
                  </>
                )}
              </div>
              <label
                htmlFor='queryImage'
                className={styles.fileUpload}>
                <span>
                  <i className='bi bi-upload'></i>
                </span>
                Upload query image for face recognition
              </label>
              <input
                id='queryImage'
                type='file'
                accept='image/jpeg, image/png, image/webp'
                onChange={handleQueryImage}
                hidden
              />
            </div>
          </div>
          <div className={styles.buttonContainer}>
            <button
              className='btn btn-primary'
              onClick={() => loadRecognizedFaces()}>
              Recognize Face
            </button>
          </div>
          {recognitionError && (
            <div className='alert alert-danger' role='alert'>
              {recognitionError}
            </div>
          )}
          {livenessDetected && (
            <div className='alert alert-success' role='alert'>
              Liveness detected!
            </div>
          )}
          <div>
            <b>Note</b>: Assume the data set as the Aadhar dataset with all the images
            <ol>
              <li>Image should be of single person only.</li>
              <li>
                Name of the file/image is considered to
                recognize the person from query image.
              </li>
              <li>
                For example, image is uploaded with file name
                neeraj.jpg, neeraj (1).jpg, vageele.jpg, etc.
                the label will show neeraj if face is matched
                from data ser in query image.
              </li>
            </ol>
            <b>Privacy Notice:</b>
            <ul>
              <li>
                Uploaded Image are not saved or collected any
                where for any purpose.The Model is running on device and no server is involved.
              </li>
            </ul>
          </div>
        </div>
        <div className={styles.videoSection}>
          <h4>Live Video Feed</h4>
          <video ref={videoRef} autoPlay muted width={500} height={500} />
        </div>
      </div>
    </>
  );
}