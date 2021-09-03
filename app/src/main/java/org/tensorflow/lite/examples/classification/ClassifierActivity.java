/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Matrix;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.os.Vibrator;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.TextureView;
import android.view.ViewStub;
import android.widget.TextView;
import android.widget.Toast;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.ArrayList;

import org.tensorflow.lite.examples.classification.customview.AutoFitTextureView;
import org.tensorflow.lite.examples.classification.customview.OverlayView;
import org.tensorflow.lite.examples.classification.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.classification.env.BorderedText;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.env.ImageUtils;
import org.tensorflow.lite.examples.classification.tflite.Classifier_Midas;
import org.tensorflow.lite.examples.classification.tflite.Classifier_Midas.Device;
import org.tensorflow.lite.examples.classification.tflite.Classifier_Midas.Model;
import org.tensorflow.lite.examples.classification.tflite.Classifier_Yolo;
import org.tensorflow.lite.examples.classification.tflite.DetectorFactory;
import org.tensorflow.lite.examples.classification.tflite.YoloV5Classifier;
import org.tensorflow.lite.examples.classification.tracking.MultiBoxTracker;

import android.widget.ImageView;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.PixelFormat;
import java.nio.ByteBuffer;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final float TEXT_SIZE_DIP = 10;
  public static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f; // 물체 인식 확률 TH

  private Bitmap rgbFrameBitmap = null;
  private long lastProcessingTimeMs;
  private Integer sensorOrientation;
  private Classifier_Midas classifier;
  private BorderedText borderedText;

  /** Input image size of the model along x axis. */
  private int imageSizeX;
  /** Input image size of the model along y axis. */
  private int imageSizeY;

  /** Yolo only */
  private static final boolean MAINTAIN_ASPECT = true;
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  private static final boolean SAVE_PREVIEW_BITMAP = false; // 비트맵 저장 여부
  private boolean computingDetection = false;;
  private long timestamp = 0;
  private YoloV5Classifier detector; // 물체 인식 클래스
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;
  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  private MultiBoxTracker tracker;
  OverlayView trackingOverlay;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    final int modelIndex = modelView.getCheckedItemPosition();
    final String modelString = modelStrings.get(modelIndex);

    try {
      detector = DetectorFactory.getDetector(getAssets(), modelString);
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    recreateClassifier(getModel(), getDevice(), getNumThreads());
    if (classifier == null) {
      LOGGER.e("No classifier on preview!");
      return;
    }

    int cropSize = detector.getInputSize();
    Log.d("DetectorActivity","input size = " + cropSize);

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropSize, cropSize,
                    sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                tracker.draw(canvas);
                if (isDebug()) {
                  tracker.drawDebug(canvas);
                }
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  // 상민 생성 함수
  protected Bitmap GraycaleToBitmap(float[] img_array, int imageSizeX, int imageSizeY) {
    float maxval = Float.NEGATIVE_INFINITY;
    float minval = Float.POSITIVE_INFINITY;
    for (float cur : img_array) {
      maxval = Math.max(maxval, cur);
      minval = Math.min(minval, cur);
    }
    float multiplier = 0;
    if ((maxval - minval) > 0) multiplier = 255 / (maxval - minval);

    int[] img_normalized = new int[img_array.length];
    for (int i = 0; i < img_array.length; ++i) {
      float val = (float) (multiplier * (img_array[i] - minval));
      img_normalized[i] = (int) val;
    }

    int width = imageSizeX; // 256
    int height = imageSizeY; // 256

    Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);

    for (int ii = 0; ii < width; ii++) //pass the screen pixels in 2 directions
    {
      for (int jj = 0; jj < height; jj++) {
        int index = (width - ii - 1) + (height - jj - 1) * width;
        if(index < img_array.length) {
          int val = img_normalized[index];
          bitmap.setPixel(ii, jj, Color.rgb(val, val, val));
        }
      }
    }

    Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 480, 480, true);

    Log.d("resizedBitmap",""+resizedBitmap.getWidth()+","+resizedBitmap.getHeight());
    return resizedBitmap;
  }

  // 추론 및 알고리즘 동작
  @Override
  protected boolean processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return false;
    }
    computingDetection = true;

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final int cropSize = Math.min(previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // 이미지 저장 (실제로는 실행이 안됨)
    //if (SAVE_PREVIEW_BITMAP) {
    //  ImageUtils.saveBitmap(croppedBitmap);
    //}

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            if (classifier != null) {
              final long startTime = SystemClock.uptimeMillis();

              // Yolo 추론 : results에 감지된 객체들의 정보가 저장됨 (getLocation으로 Box 정보 불러옴)
              final List<Classifier_Yolo.Recognition> results = detector.recognizeImage(croppedBitmap);
              Log.e("CHECK", "run: " + results.size());

              // tts 음성 출력 Demo
              int objCnt = results.size();
              if(objCnt >= 1) {
                tts.setPitch(1.5f);
                tts.setSpeechRate(1.5f);
                // editText에 있는 문장을 읽는다.
                tts.speak(""+objCnt, TextToSpeech.QUEUE_ADD, null);
                /** TTS.setPitch(float pitch) : 음성 톤 높이 설정 (배수 설정)
                 *  TTS.setSpeechRate(float speechRate) : 읽는 속도 설정 (배수 설정)
                 *
                 *  TextToSpeech.QUEUE_FLUSH : 진행중인 음성 출력을 끊고 이번 TTS의 음성 출력을 한다.
                 *  TextToSpeech.QUEUE_ADD   : 진행중인 음성 출력이 끝난 후에 이번 TTS의 음성 출력을 진행한다. */

                //vibrator.cancel();
                //vibrator.vibrate(500); // 0.5초간 진동
              }

              // Midas 추론 : img_array에 결과 이미지 저장됨
              float[] img_array = classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
              //Bitmap bitmap_Midas = GraycaleToBitmap(img_array, imageSizeX, imageSizeY); // bitmap으로 변환 (미완성)



              // 감지된 물체의 Box를 화면에 그림
              cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
              final Canvas canvas = new Canvas(cropCopyBitmap);
              final Paint paint = new Paint();
              paint.setColor(Color.RED);
              paint.setStyle(Paint.Style.STROKE);
              paint.setStrokeWidth(1.0f);

              float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
              switch (MODE) {
                case TF_OD_API:
                  minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                  break;
              }

              final List<Classifier_Yolo.Recognition> mappedRecognitions =
                      new LinkedList<Classifier_Yolo.Recognition>();

              // 임시 검출 객체(detectedObj) 리스트.
              ArrayList<DetectedObj> temp_objects = new ArrayList<>();

              for (final Classifier_Yolo.Recognition result : results) {
                // 480, 480 scale좌표를 256, 256으로 변환.
                int depth_x = (int)((result.getLocation().centerX()/480.0) * 256);
                int depth_y = (int)((result.getLocation().centerY()/480.0) * 256);
                float midas_val = img_array[(256 - depth_x - 1) + (256 - depth_y - 1) * 256];
                float disparity = 0.20f * midas_val - 20.0f;
                float distance = 119.975f * 1397.f / disparity;//baseline * focal_length / disp
                float distance_m = distance/1000.f;
                Log.d("midas", "x, y = ("+depth_x+", "+depth_y+"), val="+midas_val+", dist "+distance_m+"(m)" );

                DetectedObj temp_obj = new DetectedObj(result.getTitle(), 0,
                        result.getLocation().centerX(),
                        result.getLocation().centerY(),
                        distance_m);

                temp_objects.add(temp_obj);

                final RectF location = result.getLocation();
                if (location != null && result.getConfidence() >= minimumConfidence) {
                  canvas.drawRect(location, paint);
                  cropToFrameTransform.mapRect(location);
                  result.setLocation(location);
                  // 거리 정보 입력.
                  result.setDistance(distance_m);
                  mappedRecognitions.add(result);
                }
              }

//              for (DetectedObj obj: valid_objects)
//              {
//                // 기존 객체와 매칭하는 부분
//                obj.traceObj(temp_objects);
//              }

              for (DetectedObj temp : temp_objects)
              {
                // 매칭되지 않은 새로운 객체를 전역리스트에 추가
                if (temp.isNewItem())
                {
                  //temp.setNewItem(false);
                  valid_objects.add(temp);
                }
              }

//              for (DetectedObj obj: valid_objects)
//              {
                // 시간 지난 obj
                //if (obj.isNewItem())
                //{
//                  valid_objects.remove(obj);
//                }
//              }


              tracker.trackResults(mappedRecognitions, currTimestamp);
              trackingOverlay.postInvalidate();
              /////////////////////////////////////////////////////////

              computingDetection = false;

              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

              runOnUiThread(
                  new Runnable() {
                    @Override
                    public void run() {
                      showResultsInTexture(img_array, imageSizeX, imageSizeY); // Midas 추론 결과를 화면에 띄움
                      showCameraResolution(cropSize + "x" + cropSize);
                      showRotationInfo(String.valueOf(sensorOrientation));

                      showFrameInfo(previewWidth + "x" + previewHeight);
                      showCropInfo(imageSizeX + "x" + imageSizeY);
                      showInference(lastProcessingTimeMs + "ms");
                    }
                  });
            }
            readyForNextImage();
          }
        });
    return true;
  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_ic_camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  protected void onInferenceConfigurationChanged() {
    if (rgbFrameBitmap == null) {
      // Defer creation until we're getting camera frames.
      return;
    }
    final Device device = getDevice();
    final Model model = getModel();
    final int numThreads = getNumThreads();
    runInBackground(() -> recreateClassifier(model, device, numThreads));
  }

  // Midas classifier 재생성
  private void recreateClassifier(Model model, Device device, int numThreads) {
    if (classifier != null) {
      LOGGER.d("Closing classifier.");
      classifier.close();
      classifier = null;
    }
    if (device == Device.GPU
        && (model == Model.QUANTIZED_MOBILENET || model == Model.QUANTIZED_EFFICIENTNET)) {
      LOGGER.d("Not creating classifier: GPU doesn't support quantized models.");
      runOnUiThread(
          () -> {
            Toast.makeText(this, R.string.tfe_ic_gpu_quant_error, Toast.LENGTH_LONG).show();
          });
      return;
    }
    try {
      LOGGER.d(
          "Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads);
      Log.d("ClassifierActivity","model - "+model);
      classifier = Classifier_Midas.create(this, model, device, numThreads);
    } catch (IOException | IllegalArgumentException e) {
      LOGGER.e(e, "Failed to create classifier.");
      runOnUiThread(
          () -> {
            Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
          });
      return;
    }

    // Updates the input image size.
    imageSizeX = classifier.getImageSizeX();
    imageSizeY = classifier.getImageSizeY();
  }

  /////////////////////////////////////////////////////////////////////
  // Yolo only
  /////////////////////////////////////////////////////////////////////
  protected void updateActiveModel() {
    // Get UI information before delegating to background
    final int modelIndex = modelView.getCheckedItemPosition();
    final int deviceIndex = deviceView.getCheckedItemPosition();
    String threads = threadsTextView.getText().toString().trim();
    final int numThreads = Integer.parseInt(threads);

    handler.post(() -> {
      if (modelIndex == currentModel && deviceIndex == currentDevice
              && numThreads == currentNumThreads) {
        return;
      }
      currentModel = modelIndex;
      currentDevice = deviceIndex;
      currentNumThreads = numThreads;

      // Disable classifier while updating
      if (detector != null) {
        detector.close();
        detector = null;
      }

      // Lookup names of parameters.
      String modelString = modelStrings.get(modelIndex);
      String device = deviceStrings.get(deviceIndex);

      LOGGER.i("Changing model to " + modelString + " device " + device);

      // Try to load model.

      try {
        detector = DetectorFactory.getDetector(getAssets(), modelString);
        // Customize the interpreter to the type of device we want to use.
        if (detector == null) {
          return;
        }
      }
      catch(IOException e) {
        e.printStackTrace();
        LOGGER.e(e, "Exception in updateActiveModel()");
        Toast toast =
                Toast.makeText(
                        getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
      }

      if (device.equals("CPU")) {
        detector.useCPU();
      } else if (device.equals("GPU")) {
        detector.useGpu();
      } else if (device.equals("NNAPI")) {
        detector.useNNAPI();
      }
      detector.setNumThreads(numThreads);

      int cropSize = detector.getInputSize();
      croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

      frameToCropTransform =
              ImageUtils.getTransformationMatrix(
                      previewWidth, previewHeight,
                      cropSize, cropSize,
                      sensorOrientation, MAINTAIN_ASPECT);

      cropToFrameTransform = new Matrix();
      frameToCropTransform.invert(cropToFrameTransform);
    });
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  //@Override
  //protected void setNumThreads(final int numThreads) {
  //  runInBackground(() -> detector.setNumThreads(numThreads));
  //}
}
