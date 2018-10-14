package embeddingTraining;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class readData {
    private String entity2idAddress;
    private String relation2idAddress;
    private String trainAddress;

    public int entityNum = 0;
    public int relationNum = 0;
    public int tripleNum = 0;

    public Map<String, Integer> entity2id = new HashMap<>();  //<v, the id of v>
    public Map<Integer, String> id2entity = new HashMap<>();  //<the id of v, v>

    public Map<String, Integer> relation2id = new HashMap<>();  //<e, the id of e>
    public Map<Integer, String> id2relation = new HashMap<>();  //<the id of e, e>

    public List<int[]> trainingData = new ArrayList<>();  //[v_h, e, v_t]

    public Map<Integer, ArrayList<int[]>> totalHeadContext = new HashMap<>();  //for the context of vertex v, if <v, e, v_t> exists, headContext.get(v) includes [e, v_t]
    public Map<Integer, ArrayList<int[]>> totalTailContext = new HashMap<>();  //for the context of vertex v, if <v_h, e, v> exists, tailContext.get(v) includes [e, v_h]

    public Map<Integer, ArrayList<int[]>> headContext = new HashMap<>();  //head context after sampling
    public Map<Integer, ArrayList<int[]>> tailContext = new HashMap<>();  //tail context after sampling

    public int thresholdHead;
    public int thresholdTail;

    public readData(String inputAddress, int thresholdHead, int thresholdTail) throws IOException {
        System.out.println("--------------------------------------------------------");
        entity2idAddress = inputAddress + "\\e2id.txt";
        relation2idAddress = inputAddress + "\\r2id.txt";
        trainAddress = inputAddress + "\\train.txt";
        this.thresholdHead = thresholdHead;
        this.thresholdTail = thresholdTail;
        this.readingData();
        this.samplingContext();
//        this.localTest();  //for testing each function in this class
    }

    public void samplingContext(){
        for (int i = 0; i < entityNum; i++){
            ArrayList<int[]> tmpHeadContext;
            ArrayList<int[]> tmpTailContext;
            ArrayList<int[]> sampledHeadContext = new ArrayList<>();
            ArrayList<int[]> sampledTailContext = new ArrayList<>();

            if (totalHeadContext.containsKey(i)){
                tmpHeadContext = totalHeadContext.get(i);
                if (tmpHeadContext.size() > this.thresholdHead){
                    for (int j =0; j < this.thresholdHead; j++){
                        int tmpIndex = (int)(Math.random() * tmpHeadContext.size());
                        while (sampledHeadContext.contains(tmpHeadContext.get(tmpIndex)))
                            tmpIndex = (int)(Math.random() * tmpHeadContext.size());
                        sampledHeadContext.add(tmpHeadContext.get(tmpIndex));
                    }
                }
                else
                    sampledHeadContext = tmpHeadContext;
                this.headContext.put(i, sampledHeadContext);
            }

            if (totalTailContext.containsKey(i)){
                tmpTailContext = totalTailContext.get(i);
                if (tmpTailContext.size() > this.thresholdTail){
                    for (int j = 0; j < thresholdTail; j++){
                        int tmpIndex = (int)(Math.random() * tmpTailContext.size());
                        while (sampledTailContext.contains(tmpTailContext.get(tmpIndex)))
                            tmpIndex = (int)(Math.random() * tmpTailContext.size());
                        sampledTailContext.add(tmpTailContext.get(tmpIndex));
                    }
                }
                else
                    sampledTailContext = tmpTailContext;
                this.tailContext.put(i, sampledTailContext);

            }

        }
    }

    public void readingData() throws IOException {
        System.out.println("Reading entity2id data;");
        BufferedReader reader1 = new BufferedReader(new FileReader(new File(entity2idAddress)));
        String line;
        while ((line = reader1.readLine()) != null){
            line = line.trim();
            String[] entityAndId = line.split(" ");
            entity2id.put(entityAndId[0].trim(),Integer.parseInt(entityAndId[1].trim()));
            id2entity.put(Integer.parseInt(entityAndId[1].trim()),entityAndId[0].trim());
            entityNum++;
        }
        reader1.close();

        System.out.println("Reading relation2id data;");
        BufferedReader reader2 = new BufferedReader(new FileReader(new File(relation2idAddress)));
        while ((line = reader2.readLine()) != null){
            line = line.trim();
            String[] relationAndID = line.split(" ");
            relation2id.put(relationAndID[0].trim(),Integer.parseInt(relationAndID[1].trim()));
            id2relation.put(Integer.parseInt(relationAndID[1].trim()),relationAndID[0].trim());
            relationNum++;
        }
        reader2.close();

        System.out.println("Reading training data and retrieving context information for each vertex;");
        BufferedReader reader3 = new BufferedReader(new FileReader(new File(trainAddress)));
        while((line = reader3.readLine()) != null){
            line = line.trim();
            String[] headAndRelationAndTail = line.split("\t");
            String head = headAndRelationAndTail[0].trim();
            String relation = headAndRelationAndTail[1].trim();
            String tail = headAndRelationAndTail[2].trim();
            int[] triple = new int[3];
            triple[0] = entity2id.get(head);
            triple[1] = relation2id.get(relation);
            triple[2] = entity2id.get(tail);
            int[] tmpHeadContext = {triple[1], triple[2]};
            int[] tmpTailContext = {triple[1], triple[0]};
            if (totalHeadContext.containsKey(triple[0]))
                totalHeadContext.get(triple[0]).add(tmpHeadContext);
            else {
                ArrayList<int[]> tmpArrayList = new ArrayList<>();
                tmpArrayList.add(tmpHeadContext);
                totalHeadContext.put(triple[0], tmpArrayList);
            }
            if (totalTailContext.containsKey(triple[2]))
                totalTailContext.get(triple[2]).add(tmpTailContext);
            else{
                ArrayList<int[]> tmpArrayList = new ArrayList<>();
                tmpArrayList.add(tmpTailContext);
                totalTailContext.put(triple[2], tmpArrayList);
            }
            trainingData.add(triple);
            tripleNum++;
        }
        reader3.close();

        System.out.println("entityNum: " + entityNum + ", relationNum: " + relationNum + ", trainingTripleNum: " + tripleNum);
    }

//    public void localTest(){
//        System.out.println("testResultBegin: ");
//
//        for (Entry<String, Integer> entry1 : entity2id.entrySet()) {
//            System.out.println(entry1.getKey() + " " + entry1.getValue());
//        }
//
//        for (Entry<Integer, String> entry2 : id2entity.entrySet()) {
//            System.out.println(entry2.getKey() + " " + entry2.getValue());
//        }
//
//        for (Entry<String, Integer> entry3 : relation2id.entrySet()) {
//            System.out.println(entry3.getKey() + " " + entry3.getValue());
//        }
//
//        for (Entry<Integer, String> entry4 : id2relation.entrySet()) {
//            System.out.println(entry4.getKey() + " " + entry4.getValue());
//        }
//
//        for(int i=0; i<100; i++){
//            System.out.println(id2entity.get(trainingData.get(i)[0]) + " " + id2relation.get(trainingData.get(i)[1]) + " " + id2entity.get(trainingData.get(i)[2]));
//        }

//        String vertexForTest = "/m/027rn";
//        int testVertex = entity2id.get(vertexForTest);
//        int count1 = 0;
//        int count2 = 0;
//        ArrayList<int[]> tmpHeadArrayList;
//        tmpHeadArrayList = totalHeadContext.get(testVertex);
//        System.out.println("the head context of " + vertexForTest +":");
//        for (int[] aTmpHeadArrayList : tmpHeadArrayList) {
//            System.out.println(id2relation.get(aTmpHeadArrayList[0]) + " " + id2entity.get(aTmpHeadArrayList[1]));
//            count1++;
//        }
//        ArrayList<int[]> tmpTailArrayList;
//        tmpTailArrayList = totalTailContext.get(testVertex);
//        System.out.println("the tail context of " + vertexForTest +":");
//        for (int[] aTmpTailArrayList : tmpTailArrayList) {
//            System.out.println(id2relation.get(aTmpTailArrayList[0]) + " " + id2entity.get(aTmpTailArrayList[1]));
//            count2++;
//        }
//        System.out.println("the vertex " + vertexForTest + " has " + count1 + " head context and " + count2 +" tail context");
//
//        String vertexForTest = "/m/027rn";
//        int testVertex = entity2id.get(vertexForTest);
//        int count1 = 0;
//        int count2 = 0;
//        ArrayList<int[]> tmpHeadArrayList;
//        tmpHeadArrayList = headContext.get(testVertex);
//        System.out.println("the head context of " + vertexForTest +":");
//        for (int[] aTmpHeadArrayList : tmpHeadArrayList) {
//            System.out.println(id2relation.get(aTmpHeadArrayList[0]) + " " + id2entity.get(aTmpHeadArrayList[1]));
//            count1++;
//        }
//        ArrayList<int[]> tmpTailArrayList;
//        tmpTailArrayList = tailContext.get(testVertex);
//        System.out.println("the tail context of " + vertexForTest +":");
//        for (int[] aTmpTailArrayList : tmpTailArrayList) {
//            System.out.println(id2relation.get(aTmpTailArrayList[0]) + " " + id2entity.get(aTmpTailArrayList[1]));
//            count2++;
//        }
//        System.out.println("the vertex " + vertexForTest + " has " + count1 + " head context and " + count2 +" tail context");
//
//        System.out.println("testResultEnd ");
//    }
}
