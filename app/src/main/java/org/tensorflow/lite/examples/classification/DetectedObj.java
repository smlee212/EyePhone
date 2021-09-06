package org.tensorflow.lite.examples.classification;

import android.util.Log;
import java.util.ArrayList;

public class DetectedObj {
    private String className; // 객체 종류
    private int id = -1; // 객체 고유 번호

    private int ALSize = 0;
    private ArrayList<Float> xPos = new ArrayList<>(); // 위치 좌표 x (이미지 상에서는 y로 쓰임)
    private ArrayList<Float> yPos = new ArrayList<>(); // 위치 좌표 y (이미지 상에서는 x로 쓰임)
    private ArrayList<Float> depth = new ArrayList<>(); // 객체와의 거리

    private float h = 0; // 위치 좌표로부터 중심점까지의 거리
    private float dx = 0; // 방향 벡터 dx
    private float dy = 0; // 방향 벡터 dy

    private int state = 0; // 새롭게 검출된 객체인가
    // 처음으로 발견된 객체이면 0
    // 이전 객체로부터 추적되어 정보를 갱신시켜줄 객체이면 1 (임시 리스트의 원소 기준)
    // 2번 이상 추적된 객체이면 2 (전역 리스트의 원소 기준)
    // 검출이 안되는 상태이면 3

    private ArrayList<Long> Time = new ArrayList<>(); // 프레임 간격 시간들 모음
    //private float totalTime = 0; // 정보 갱신을 위한 총 시간 (임계값보다 클 경우 갱신해주는 용도)
    //private float notDetectedTime = 0; // 객체가 연속적으로 검출되지 않은 시간


    // 추적 함수
    public void traceObj(ArrayList<DetectedObj> obj, long time){
        double minDist = 10000.0;
        int idx = -1;
        boolean isDetected = false;
        state = 2;

        // 검출된 객체가 없을 경우
        if(obj.size()==0){
            // 정보 갱신 후 return
            state = 3;
            return;
        }
        // 검출된 객체가 있을 경우
        else {
            for (int i = 0; i < obj.size(); i++) { // 검출된 모든 객체와 비교
                DetectedObj temp = obj.get(i); // 임시 리스트의 state는 0 또는 1이다.
                if (className != temp.getClassName()) continue; // 다른 클래스일 때
                if (temp.getState() == 1) continue; // 이미 추적이 된 물체일 때

                double d = 0;
                d = Math.pow(xPos.get(ALSize-1) - temp.getX(), 2.0) + Math.pow(yPos.get(ALSize-1) - temp.getY(), 2.0);
                d = Math.sqrt(d);

                if (d < minDist) { // 가장 가까운 객체 판별
                    isDetected = true;
                    minDist = d;
                    idx = i;
                }
            }
        }

        // 검출된 동일한 클래스가 없을 경우
        if(isDetected == false) {
            state = 3;
            return;
        }
        // 검출된 동일한 클래스가 있을 경우 - 벡터 계산
        else {
            DetectedObj temp = (DetectedObj) obj.get(idx); // 가장가까운 객체 선택
            temp.setState(1);
            float tx = temp.getX();
            float ty = temp.getY();
            float td = temp.getD();
            dx = tx - xPos.get(ALSize-1);
            dx /= time - Time.get(ALSize-1);
            dx *= 300;
            dy = ty - yPos.get(ALSize-1);
            dy /= time - Time.get(ALSize-1);
            dy *= 300;
            xPos.add(tx);
            yPos.add(ty);
            depth.add(td);
            h = temp.getH();
            Time.add(time);
            ALSize++;
        }
    }

    public boolean refresh(long currentTime){
        // 새롭게 추가된 객체일 경우
        if(state == 0) {
            return true;
        }

        //float fronttTime = Time.get(0); // 첫번째 값 반환
        float totalTime = currentTime - Time.get(0);

        // 최근 정보 갱신 작업 (임계 시간을 넘긴 경우 과거 정보 제거)
        while(totalTime >= 2000) {
            if(ALSize > 1)
            {
                Time.remove(0); // 첫번째 값 pop
                xPos.remove(0);
                yPos.remove(0);
                depth.remove(0);
                ALSize--;
            }
            else // 객체가 마지막으로 검출된 후 임계 시간이 자났기에 return 후 제거시켜준다.
            {
                return false;
            }
            totalTime = currentTime - Time.get(0);
        }
        return true;
    }

    public void showInfo(long t){
        Log.d("obj", "\t\tclass : "+String.format("%20s",className)+"(id:"+id+",state:"+state+", size:"+ALSize+") "
                                        +" - (x,y)=("+yPos.get(ALSize-1)+","+xPos.get(ALSize-1)+")"
                                        +", (dx,dy)=("+dy+","+dx+")"
                                        +", time="+(t-Time.get(0))+"ms"+", h="+h);
    }

    //////////////////////////////////////////////////////////////////////////////////

    public DetectedObj(String className, float x, float y, float d, float h, long t) {
        this.className = className;
        this.xPos.add(x);
        this.yPos.add(y);
        this.depth.add(d);
        this.h = h;
        this.Time.add(t);
        ALSize = 1;
    }

    public String getClassName() {
        return className;
    }

    public int getId() {
        return id;
    }

    public ArrayList<Float> getxPos() { return xPos; }

    public float getX() { return xPos.get(ALSize-1); }

    public ArrayList<Float> getyPos() { return yPos; }

    public float getY() { return yPos.get(ALSize-1); }

    public ArrayList<Float> getDepth() { return depth; }

    public float getD() { return depth.get(ALSize-1); }

    public float getDx() { return dx; }

    public float getDy() { return dy; }

    public float getH() { return h; }

    public int getState() { return state; }

    public String getInfo(){ return "id: "+id+", className: "+className+", pos: ("+xPos+", "+yPos+"), distance: "+depth; }

    public int getALSize() {return ALSize; }

    //public float getNotDetectedTime() { return notDetectedTime; }

    public void setId(int i) { id = i; }

    public void setState(int i) { state = i; }
}
