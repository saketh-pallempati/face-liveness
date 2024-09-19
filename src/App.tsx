import { useCallback, useEffect, useRef, useState } from 'react';
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
} from 'face-api.js';

import styles from './faceRecognition.module.scss';

const MODEL_URL = '/models';
const FACE_MATCHER_THRESHOLD = 0.6;
const VALID_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export default function FaceRecognition() {
  const [dataSetImages, setDataSetImages] = useState([]);
  const [faceMatcher, setFaceMatcher] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loaderMsg, setLoaderMsg] = useState('');
  const [recognitionError, setRecognitionError] = useState('');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const refImgElements = useRef([]);

  const addImageRef = (index, ref) => {
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

  const startVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Error accessing the camera", err);
      setRecognitionError('Failed to access the camera. Please check your permissions.');
    }
  };

  const detectFaces = async () => {
    if (!videoRef.current || !canvasRef.current || !faceMatcher) return;

    const detections = await detectAllFaces(videoRef.current)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const canvas = canvasRef.current;
    matchDimensions(canvas, videoRef.current);

    const resizedDetections = resizeResults(detections, {
      width: videoRef.current.width,
      height: videoRef.current.height,
    });

    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);

    resizedDetections.forEach(detection => {
      const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
      const drawBox = new draw.DrawBox(detection.detection.box, {
        label: bestMatch.toString(),
      });
      drawBox.draw(canvas);
    });

    requestAnimationFrame(detectFaces);
  };

  const setImagesForRecognition = (event) => {
    const files = Array.from(event.target.files || []);

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
  }, []);

  useEffect(() => {
    if (dataSetImages?.length > 0) {
      processImagesForRecognition();
    }
  }, [dataSetImages, processImagesForRecognition]);

  useEffect(() => {
    if (faceMatcher) {
      startVideo();
    }
  }, [faceMatcher]);

  useEffect(() => {
    if (videoRef.current && faceMatcher) {
      videoRef.current.addEventListener('play', detectFaces);
    }
    return () => {
      if (videoRef.current) {
        videoRef.current.removeEventListener('play', detectFaces);
      }
    };
  }, [faceMatcher]);

  return (
    <>
      {isLoading && <div>{loaderMsg}</div>}
      <div className={`container ${styles.container}`}>
        <div className={`${styles.imageSection} ${styles.multiImageSection}`}>
          <div className={styles.twoSectionPreview}>
            <div className={styles.dataSetSection}>
              <h4>Original Data</h4>
              <div>
                {dataSetImages?.map((image, index) => (
                  <div className={styles.imageArea} key={`data-set-${index}`}>
                    <img
                      ref={(imageRef) => addImageRef(index, imageRef)}
                      src={image.src}
                      alt={image.name}
                      width={100}
                      height={100}
                    />
                    <span>{image.name}</span>
                  </div>
                ))}
              </div>
              <label htmlFor="multiFileSelect" className={styles.fileUpload}>
                <span><i className="bi bi-upload"></i></span>
                Upload image data set for face recognition
              </label>
              <input
                id="multiFileSelect"
                type="file"
                onChange={setImagesForRecognition}
                multiple
                accept="image/jpeg, image/png, image/webp"
                hidden
              />
            </div>
            <div className={styles.queryImageSection}>
              <h4>Video Feed</h4>
              <div className={styles.videoContainer}>
                <video
                  ref={videoRef}
                  width={500}
                  height={500}
                  autoPlay
                  muted
                  className={styles.video}
                />
                <canvas
                  ref={canvasRef}
                  className={styles.canvas}
                  width={500}
                  height={500}
                />
              </div>
            </div>
          </div>
          {recognitionError && (
            <div className="alert alert-danger" role="alert">
              {recognitionError}
            </div>
          )}
          <div>
            <b>Note</b>: Assume the data set as the Aadhar dataset with all the images
            <ol>
              <li>Image should be of single person only.</li>
              <li>Name of the file/image is considered to recognize the person from video feed.</li>
              <li>For example, image is uploaded with file name neeraj.jpg, neeraj (1).jpg, vageele.jpg, etc. the label will show neeraj if face is matched from data set in video feed.</li>
            </ol>
            <b>Privacy Notice:</b>
            <ul>
              <li>Uploaded Images and video feed are not saved or collected anywhere for any purpose. The model is running on device and no server is involved.</li>
            </ul>
          </div>
        </div>
      </div>
    </>
  );
}