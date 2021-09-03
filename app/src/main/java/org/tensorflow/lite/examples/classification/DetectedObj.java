package org.tensorflow.lite.examples.classification;

import android.util.Log;
import java.util.ArrayList;

public class DetectedObj {
    private String className; // 객체 종류
    private int id = -1; // 객체 고유 번호

    private int ALSize = 0;
    // 해당 변수들은 ArrayList<Float>로 변환하여
    // 검출된 프레임마다 정보를 저장해야한다.
    // 현재 간단한 구현을 위해 단일 변수로 설정한 것이므로 추후에 ArrayList로 변경해주어야 함
    private ArrayList<Float> xPos = new ArrayList<>(); // 위치 좌표 x
    private ArrayList<Float> yPos = new ArrayList<>(); // 위치 좌표 y
    private ArrayList<Float> depth = new ArrayList<>(); // 객체와의 거리

    // 이 또한 ArrayList로 변경해서 사용해야 함
    // 하지만 위의 정보들과는 List의 크기가 1개 작은것을 인지해야함
    private float dx = 0; // 방향 벡터 dx
    private float dy = 0; // 방향 벡터 dy

    private int state = 0; // 새롭게 검출된 객체인가
    // 처음으로 발견된 객체이면 0
    // 이전 객체로부터 추적되어 정보를 갱신시켜줄 객체이면 1 (임시 리스트의 원소 기준)
    // 2번 이상 추적된 객체이면 2 (전역 리스트의 원소 기준)

    private ArrayList<Long> Time = new ArrayList<>(); // 프레임 간격 시간들 모음
    //private float totalTime = 0; // 정보 갱신을 위한 총 시간 (임계값보다 클 경우 갱신해주는 용도)
    //private float notDetectedTime = 0; // 객체가 연속적으로 검출되지 않은 시간



    // 추적 함수
    public void traceObj(ArrayList<DetectedObj> obj){
        double minDist = 10000.0;
        int idx = -1;
        boolean isDetected = false;
        state = 2;

        // 검출된 객체가 없을 경우
        if(obj.size()==0){
            // 정보 갱신 후 return
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
            return;
        }
        // 검출된 동일한 클래스가 있을 경우 - 벡터 계산
        else {
            DetectedObj temp = (DetectedObj) obj.get(idx); // 가장가까운 객체 선택
            float tx = temp.getX();
            float ty = temp.getY();
            float td = temp.getD();
            dx = tx - xPos.get(ALSize-1);
            dy = ty - yPos.get(ALSize-1);
            xPos.add(tx);
            yPos.add(ty);
            depth.add(td);
            ALSize++;
        }
    }


    public boolean refresh(long time){
        // 새롭게 추가된 객체일 경우
        if(state == 0) {
            return true;
        }

        float fronttTime = Time.get(0); // 첫번째 값 반환
        float totalTime = time - fronttTime;

        // 최근 정보 갱신 작업
        if(totalTime >= 0) { // TH_TIME은 최근 2초(와 같은 임계값)
            if(ALSize > 1) {
                Time.remove(0); // 첫번째 값 pop
                xPos.remove(0);
                yPos.remove(0);
                depth.remove(0);
                ALSize--;
                Time.add(time);
            }
            else{
                return false;
            }
        }

        return true;
    }

    public void showInfo(){
        Log.d("DetectedObject", "class : "+className+" (id:"+id+",state:"+state+")");

        Log.d("DetectedObject", "\tlocation : ("+xPos.get(ALSize-1)+","+yPos.get(ALSize-1)+")");
        Log.d("DetectedObject", "\tsize="+ALSize+", (dx,dy) = "+dx+","+dy+")");
    }

    //////////////////////////////////////////////////////////////////////////////////

    public DetectedObj(String className, float x, float y, float d, long t) {
        this.className = className;
        this.xPos.add(x);
        this.yPos.add(y);
        this.depth.add(d);
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

    public float getDx() {
        return dx;
    }

    public float getDy() {
        return dy;
    }

    public int getState() {
        return state;
    }

    public String getInfo(){
        return "id: "+id+", className: "+className+", pos: ("+xPos+", "+yPos+"), distance: "+depth;
    }

    public int getALSize() {return ALSize; }

    //public float getNotDetectedTime() { return notDetectedTime; }

    public void setId(int i) { id = i; }

    public void setState(int i) { state = i; }
}
