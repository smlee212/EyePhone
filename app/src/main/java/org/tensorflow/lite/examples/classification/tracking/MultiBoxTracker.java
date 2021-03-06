/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.classification.tracking;

import static java.lang.Math.PI;
import static java.lang.Math.atan2;
import static java.lang.Math.cos;
import static java.lang.Math.sin;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Matrix;
import android.graphics.Path;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.text.TextUtils;
import android.util.Pair;
import android.util.TypedValue;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import org.tensorflow.lite.examples.classification.env.BorderedText;
import org.tensorflow.lite.examples.classification.env.ImageUtils;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier_Yolo.Recognition;

/** A tracker that handles non-max suppression and matches existing objects to new detections. */
public class MultiBoxTracker {
  private static final float TEXT_SIZE_DIP = 15;
  private static final float MIN_SIZE = 16.0f;
  private static final int[] COLORS = {
          Color.BLUE,
          //Color.RED,
          Color.GREEN,
          Color.YELLOW,
          Color.CYAN,
          Color.MAGENTA,
          Color.WHITE,
          Color.parseColor("#55FF55"),
          Color.parseColor("#FFA500"),
          Color.parseColor("#FF8888"),
          Color.parseColor("#AAAAFF"),
          Color.parseColor("#FFFFAA"),
          Color.parseColor("#55AAAA"),
          Color.parseColor("#AA33AA"),
          Color.parseColor("#0D0068")
  };
  final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();
  private final Logger logger = new Logger();
  private final Queue<Integer> availableColors = new LinkedList<Integer>();
  private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();
  private final Paint boxPaint = new Paint();
  private final float textSizePx;
  private final BorderedText borderedText;
  private Matrix frameToCanvasMatrix;
  private int frameWidth;
  private int frameHeight;
  private int sensorOrientation;

  public MultiBoxTracker(final Context context) {
    for (final int color : COLORS) {
      availableColors.add(color);
    }

    boxPaint.setColor(Color.RED);
    boxPaint.setStyle(Style.STROKE);
    boxPaint.setStrokeWidth(5.0f);
    boxPaint.setStrokeCap(Cap.ROUND);
    boxPaint.setStrokeJoin(Join.ROUND);
    boxPaint.setStrokeMiter(100);

    textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
  }

  public synchronized void setFrameConfiguration(
          final int width, final int height, final int sensorOrientation) {
    frameWidth = width;
    frameHeight = height;
    this.sensorOrientation = sensorOrientation;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = detection.second;
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }
  }

  public synchronized void trackResults(final List<Recognition> results, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(results);
  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void draw(final Canvas canvas) {
    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
            Math.min(
                    canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                    canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
            ImageUtils.getTransformationMatrix(
                    frameWidth,
                    frameHeight,
                    (int) (multiplier * (rotated ? frameHeight : frameWidth)),
                    (int) (multiplier * (rotated ? frameWidth : frameHeight)),
                    sensorOrientation,
                    false);

    Paint paint_c=new Paint();
    paint_c.setColor(Color.RED);
    paint_c.setStrokeWidth(5);
    paint_c.setStyle(Paint.Style.STROKE);
    //canvas.drawCircle(240f*(1080f/480f), (int)(640f)*(1080f/480f),(int)(240f)*(1080f/480f),paint_c);
    RectF rect_c = new RectF(0f*(1080f/480f), (640f-240f)*(1080f/480f), 480f*(1080f/480f), (640f+240f)*(1080f/480f));

    canvas.drawArc(rect_c, 180, 180,true, paint_c);

    //Paint paint_l = new Paint();
    //paint_l.setColor(Color.BLUE);
    //paint_l.setStrokeWidth(10);
    //canvas.drawLine(0, (480+80)*(1080f/480f), 480*(1080f/480f), (480+80)*(1080f/480f), paint_l);
    //canvas.drawLine(0, (0+80)*(1080f/480f), 480*(1080f/480f), (0+80)*(1080f/480f), paint_l);



    for (final TrackedRecognition recognition : trackedObjects) {
      final RectF trackedPos = new RectF(recognition.location);

      getFrameToCanvasMatrix().mapRect(trackedPos);
      boxPaint.setColor(recognition.color);
      boxPaint.setStrokeWidth(5.f);
      boxPaint.setTextSize(50f);
      // ??????
      DashPathEffect dashPath = new DashPathEffect(new float[]{15,15}, 2);
      boxPaint.setPathEffect(dashPath);

      float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 25.0f;
      canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

      boxPaint.setPathEffect(null);

      String rangeDistance;
      if (recognition.distance < 3) {
        rangeDistance = "1~3[m]";
        RectF tempRectF = new RectF(trackedPos);
        tempRectF.set(trackedPos.left-10, trackedPos.top-10, trackedPos.right+10, trackedPos.bottom+10);
        Paint tempPaint = new Paint(boxPaint);
        tempPaint.setColor(Color.RED);
        canvas.drawRoundRect(tempRectF, cornerSize, cornerSize, tempPaint);
      }
      else if (recognition.distance < 6) {
        rangeDistance = "3~6[m]";
      }
      else if (recognition.distance < 10) {
        rangeDistance = "6~10[m]";
      }
      else {
        rangeDistance = "10[m]~";
      }


      final String labelString =
              !TextUtils.isEmpty(recognition.title)
                      ? String.format("%s %s", recognition.title, rangeDistance)
                      : String.format("%sf", rangeDistance);

//      final String labelString =
//              !TextUtils.isEmpty(recognition.title)
//                      ? String.format("%s %.2f", recognition.title, (100 * recognition.detectionConfidence))
//                      : String.format("%.2f", (100 * recognition.detectionConfidence));

      //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
      // labelString);

      //final String labelDistance = String.format("%.2f(m)", recognition.distance);

      borderedText.drawText(
              canvas, trackedPos.left + cornerSize, trackedPos.top, labelString, boxPaint);

      ////// -> //////
      // ?????????????????? //
      ////// <- //////
      boxPaint.setStrokeWidth(8.f);
      //drawArrow(boxPaint, canvas, trackedPos.centerX(), trackedPos.centerY(),
      //        trackedPos.centerX()-recognition.dy*1, trackedPos.centerY()+recognition.dx*1); //
      drawArrow(boxPaint, canvas, trackedPos.centerX(), trackedPos.centerY()+trackedPos.height()/2,
              trackedPos.centerX()+recognition.dy*1, trackedPos.centerY()+trackedPos.height()/2+recognition.dx*1); //

    }
  }

  private void processResults(final List<Recognition> results) {
    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }
      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
              "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
    }

    trackedObjects.clear();
    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    for (final Pair<Float, Recognition> potential : rectsToTrack) {
      // ????????? ??????????????? ???????????? ??????
      final TrackedRecognition trackedRecognition = new TrackedRecognition();
      trackedRecognition.detectionConfidence = potential.first;
      trackedRecognition.location = new RectF(potential.second.getLocation());
      trackedRecognition.title = potential.second.getTitle();
      trackedRecognition.distance = potential.second.getDistance();
      trackedRecognition.dx = potential.second.getDx();
      trackedRecognition.dy = potential.second.getDy();
      //trackedRecognition.title = "hello world!";
//      trackedRecognition.color = COLORS[trackedObjects.size() % COLORS.length];
      trackedRecognition.color = COLORS[potential.second.getDetectedClass() % COLORS.length];
      trackedObjects.add(trackedRecognition);

//      if (trackedObjects.size() >= COLORS.length) {
//        break;
//      }
    }
  }

  private static class TrackedRecognition {
    RectF location;
    float detectionConfidence;
    float distance;
    float dx, dy;
    int color;
    String title;
  }


  private void drawArrow(Paint paint, Canvas canvas, float from_x, float from_y, float to_x, float to_y)
  {
    float angle, anglerad, radius, lineangle;

    //values to change for other appearance *CHANGE THESE FOR OTHER SIZE ARROWHEADS*
    radius=30f;
    angle=35f;

    //some angle calculations
    anglerad = (float) (PI*angle/180.0f);
    lineangle = (float) (atan2(to_y-from_y,to_x-from_x));

    //tha line
    canvas.drawLine(from_x,from_y,to_x,to_y,paint);

    //tha triangle
    Path path = new Path();
    path.setFillType(Path.FillType.EVEN_ODD);
    path.moveTo(to_x, to_y);
    path.lineTo((float)(to_x-radius*cos(lineangle - (anglerad / 2.0))),
            (float)(to_y-radius*sin(lineangle - (anglerad / 2.0))));
    path.lineTo((float)(to_x-radius*cos(lineangle + (anglerad / 2.0))),
            (float)(to_y-radius*sin(lineangle + (anglerad / 2.0))));
    path.close();

    canvas.drawPath(path, paint);
  }

}

