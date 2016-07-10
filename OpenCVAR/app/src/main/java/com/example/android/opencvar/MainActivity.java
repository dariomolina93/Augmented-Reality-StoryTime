package com.example.android.opencvar;


import android.app.ActionBar;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public final class MainActivity extends AppCompatActivity implements
        CameraBridgeViewBase.CvCameraViewListener2 {

    @Override
    protected void onCreate(Bundle savedInstanceState){
        Log.i(TAG, "called onCreate");

        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        requestWindowFeature(Window.FEATURE_NO_TITLE);// hide statusbar of Android
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.HelloVisionView);

        //Set the view as visible
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        //Register your activity as the callback object to handle //camera frames
        mOpenCvCameraView.setCvCameraViewListener(this);

    }//end onCreate

    //A Tag to filter the log messages
    private static final String  TAG = "Example::HelloVisionWorld::Activity";

    //A class used to implement the interaction between OpenCV and the //device camera.
    private CameraBridgeViewBase mOpenCvCameraView;
    //This is the callback object used when we initialize the OpenCV //library asynchronously
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        //This is the callback method called once the OpenCV //manager is connected
        public void onManagerConnected(int status) {
            switch (status) {
                //Once the OpenCV manager is successfully connected we can enable the camera interaction with the defined OpenCV camera view
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };//end BaseLoaderCallback

    @Override
    public void onResume(){

        super.onResume();

        //Call the async initialization and pass the callback object we //created later, and chose which version of OpenCV library to //load. Just make sure that the OpenCV manager you installed //supports the version you are trying to load.
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
    }//end onResume

    private Mat mRgba;
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
    }// end onCameraViewStarted

    @Override
    public void onCameraViewStopped() {

    }//end onCameraViewStopped

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat grayFrame = inputFrame.gray();
        Mat colorFrame = inputFrame.rgba();
        org.opencv.core.Size s = new Size(5, 5);

        Imgproc.GaussianBlur(grayFrame, grayFrame, s, 0);

        //Imgproc.adaptiveThreshold(grayFrame, grayFrame, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 25, 5);
        Imgproc.Canny(grayFrame, grayFrame, 50, 200);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        MatOfPoint2f approxCurve = new MatOfPoint2f();

        MatOfPoint approxContour = new MatOfPoint();

        Imgproc.findContours(grayFrame, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        for(int contourIdx = 0; contourIdx < contours.size(); contourIdx++){
            //Convert contours(contourIdx) from MatOfPoint to matOfPoint2f
            MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(contourIdx).toArray());
            double approx = Imgproc.arcLength(contour2f, true) * 0.02;
            Imgproc.approxPolyDP(contour2f, approxCurve, approx, true);

            approxCurve.convertTo(approxContour, CvType.CV_32S);
            if(approxContour.size().height == 4) {
                Imgproc.drawContours(colorFrame, contours, contourIdx, new Scalar(255, 0, 0), -1);
            }
        }
        //We're returning the colored frame as is to be rendered on //thescreen.
        //where we will be working
        return colorFrame;
    }// end onCameraFrame

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView != null){
            mOpenCvCameraView.disableView();
        }
    }



}
