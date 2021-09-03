package org.tensorflow.lite.examples.classification;

import java.util.ArrayList;

public class DetectedObj {
    private final String className; // 객체 종류
    private final int id; // 객체 고유 번호

    // 해당 변수들은 ArrayList<Float>로 변환하여
    // 검출된 프레임마다 정보를 저장해야한다.
    // 현재 간단한 구현을 위해 단일 변수로 설정한 것이므로 추후에 ArrayList로 변경해주어야 함
    private float xPos; // 위치 좌표 x
    private float yPos; // 위치 좌표 y
    private float depth; // 객체와의 거리

    // 이 또한 ArrayList로 변경해서 사용해야 함
    // 하지만 위의 정보들과는 List의 크기가 1개 작은것을 인지해야함
    private float dx = 0; // 방향 벡터 dx
    private float dy = 0; // 방향 벡터 dy

    private boolean newItem = true; // 새롭게 검출된 객체인가 (추적된 객체이면 false가 된다.)

    private ArrayList<Float> Time = new ArrayList<>(); // 프레임 간격 시간들 모음
    //11, 13
    private float totalTime = 0; // 정보 갱신을 위한 총 시간 (임계값보다 클 경우 갱신해주는 용도)
    private float notDetectedTime = 0; // 객체가 연속적으로 검출되지 않은 시간



    // 추적 함수
    public void traceObj(DetectedObj[] obj, float time){
        double minDist = 10000.0;
        int idx = -1;
        boolean isDetected = false;

        // 검출된 객체가 없을 경우
        if(obj.length==0){
            // 정보 갱신 후 return
            return;
        }
        // 검출된 객체가 있을 경우
        else {
            for (int i = 0; i < obj.length; i++) { // 검출된 모든 객체와 비교
                DetectedObj temp = (DetectedObj) obj[i];
                if (className != temp.getClassName()) continue; // 다른 클래스일 때
                if (!temp.isNewItem()) continue; // 이미 추적이 된 물체일 때

                double d = 0;
                d = Math.pow(xPos - temp.getxPos(), 2.0) + Math.pow(yPos - temp.getyPos(), 2.0);
                d = Math.sqrt(d);

                if (d < minDist) { // 가장 가까운 객체 판별
                    isDetected = true;
                    minDist = d;
                    idx = i;
                }
            }
        }

        // 최근 정보 갱신 작업
        totalTime += time;
        Time.add(time);
        if(totalTime >= 0) { // TH_TIME은 최근 2초(와 같은 임계값)
            float fronttTime = Time.get(0); // 첫번째 값 반환
            Time.remove(0); // 첫번째 값 pop
            if(totalTime == fronttTime){ // 만약 최근에 감지되었을 때와의 시간 간격이 전부였다면
                // 갱신을 해주지 않는다.
            }
            else { // 모든 정보들을 갱신한다. ArrayList로 선언된 것들을 갱신해주면 된다.
                totalTime -= fronttTime;

            }
        }

        // 검출된 동일한 클래스가 없을 경우
        if(isDetected == false) {
            notDetectedTime += time;
            if(notDetectedTime >= 0) { // TH_TIME은 최근 2초(와 같은 임계값)
                // 임계 시간동안 검출되지 않았으므로 전역 리스트에서 제거해줘야 한다.
                newItem = true; // 이를 통해 추후에 걸러낼 수 있다.
            }
        }
        // 검출된 동일한 클래스가 있을 경우 - 벡터 계산
        else {
            notDetectedTime = 0;
            DetectedObj temp = (DetectedObj) obj[idx]; // 가장가까운 객체 선택
            dx = temp.getxPos() - xPos;
            dy = temp.getyPos() - yPos;
            temp.newItem = false;
        }
    }




    public DetectedObj(String className, int id, float xPos, float yPos, float depth) {
        this.className = className;
        this.id = id;

        this.xPos = xPos;
        this.yPos = yPos;
        this.depth = depth;
    }

    public String getClassName() {
        return className;
    }

    public int getId() {
        return id;
    }

    public float getxPos() {
        return xPos;
    }

    public float getyPos() {
        return yPos;
    }

    public float getDepth() {
        return depth;
    }

    public float getDx() {
        return dx;
    }

    public float getDy() {
        return dy;
    }

    public boolean isNewItem() {
        return newItem;
    }

    public String getInfo(){
        return "id: "+id+", className: "+className+", pos: ("+xPos+", "+yPos+"), distance: "+depth;
    }
}
