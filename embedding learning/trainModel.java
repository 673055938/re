package embeddingTraining;

import java.io.*;
import java.util.ArrayList;
import java.util.Map;
import java.util.Date;
import java.text.SimpleDateFormat;


public class trainModel {
    static String inputAddress = "D:\\ISWC实验\\data\\myData";  //this folder should include "\\e2id.txt", "\\r2id.txt", and "\\train.txt"
    static String outputAddress = "D:\\ISWC实验\\data\\myData"; //output data includes "\\entity2vector.txt" and "\\relation2vector.txt", elements are separated by "\t"
    static String preVectorsAddress = "D:\\ISWC实验\\data\\myData"; //this folder should include "\\entity2vector.txt" and "\\relation2vector.txt"
    static boolean preOrRaw = true; //if "false", learning from the beginning; if "true", learning based on pre-learned vectors
    static int outputFreq = 10; //output the learned vectors per outputFrq epochs
    static int n = 50; // dimension of embedding vectors
    static double rate = 0.001;  // learning ra te
    static int epochNum = 200;  //number of epoch
    static int batchNum = 2;  //number of batches
    static int margin = 1;

    static double[][] entityVec;  //embedding vectors of vertices
    static double[][] relationVec;  //embedding vectors of edges

    static double[][] tmpEntityVec;
    static double[][] tmpRelationVec;

    static double loss; //loss cost during each epoch

    static int thresholdHead = 10;  //threshold of head context for each vertex
    static int thresholdTail = 10;  //threshold of tail context for each vertex

    public static void main(String []args) throws IOException {
        trainModel trainModel = new trainModel();
        trainModel.beginningStatement();
        readData readData = new readData(inputAddress, thresholdHead, thresholdTail);
        trainModel.initVectors(preOrRaw, preVectorsAddress, readData.entityNum,readData.relationNum);
        train(readData.entityNum, readData.relationNum,readData.headContext, readData.tailContext);
        trainModel.enddingStatement();
    }

    public static void train(int entityNum, int relationNum, Map<Integer, ArrayList<int[]>> headContext, Map<Integer, ArrayList<int[]>> tailContext) throws FileNotFoundException, UnsupportedEncodingException {
        System.out.println("--------------------------------------------------------");
        System.out.println("training...");
        int batchSize = entityNum/batchNum;
        System.out.println("the batchSize is: " + batchSize);

        int outputCount = 0;
        File outPath = new File(outputAddress);
        if(!(outPath.exists() && outPath.isDirectory())){
            outPath.mkdir();
            System.out.println("the output directory has been created");
        }
        else{
            File outEntityVecFile = new File(outputAddress + "\\entity2vector.txt");
            File outRelationVecFile = new File(outputAddress + "\\relation2vector.txt");
            if (outEntityVecFile.exists()){
                outEntityVecFile.delete();
                System.out.println("the existing entity vector file: " + outputAddress + "\\entity2vector.txt has been deleted");
            }
            if (outRelationVecFile.exists()){
                outRelationVecFile.delete();
                System.out.println("the existing relation vector file: " + outputAddress + "\\relation2vector.txt has been deleted");
            }
        }

        tmpEntityVec = new double[entityNum][n];
        tmpRelationVec = new double[relationNum][n];

        for (int epoch = 0; epoch < epochNum; epoch++){
            loss = 0;

            for (int batch = 0; batch < batchNum; batch++){

                for (int i = 0; i < entityNum; i++)
                    for (int j = 0; j < n; j++)
                        tmpEntityVec[i][j] = entityVec[i][j];
                for (int i = 0; i < relationNum; i++)
                    for (int j = 0; j < n; j++)
                        tmpRelationVec[i][j] = relationVec[i][j];

                for (int k = 0; k < batchSize; k++){
                    int trainingVertex = (int)(Math.random() * entityNum);  //current focused vertex
                    while ((!headContext.containsKey(trainingVertex)) && (!tailContext.containsKey(trainingVertex)))
                        trainingVertex = (int)(Math.random() * entityNum);

                    ArrayList<int[]> headContextOfTrainingVertex = null;
                    ArrayList<int[]> tailContextOfTrainingVertex = null;

                    if (headContext.containsKey(trainingVertex))
                        headContextOfTrainingVertex = headContext.get(trainingVertex);
                    if (tailContext.containsKey(trainingVertex))
                        tailContextOfTrainingVertex = tailContext.get(trainingVertex);

                    int negativeVertex = (int)(Math.random() * entityNum);  //pick randomly up an negative vertex
                    while ((!headContext.containsKey(negativeVertex)) && (!tailContext.containsKey(negativeVertex)))
                        negativeVertex = (int)(Math.random() * entityNum);

                    ArrayList<int[]> headContextOfNegativeVertex = null;
                    ArrayList<int[]> tailContextOfNegativeVertex = null;

                    if (headContext.containsKey(negativeVertex))
                        headContextOfNegativeVertex = headContext.get(negativeVertex);
                    if (tailContext.containsKey(negativeVertex))
                        tailContextOfNegativeVertex = tailContext.get(negativeVertex);

                    while (!(arrayListCompare(headContextOfTrainingVertex,headContextOfNegativeVertex) && arrayListCompare(tailContextOfTrainingVertex, tailContextOfNegativeVertex))){
                        negativeVertex = (int)(Math.random() * entityNum);  ////change the negative vertex if the existing one shares common context information with the training vertex
                        while ((!headContext.containsKey(negativeVertex)) && (!tailContext.containsKey(negativeVertex)))
                            negativeVertex = (int)(Math.random() * entityNum);

                        headContextOfNegativeVertex = null;
                        tailContextOfNegativeVertex = null;

                        if (headContext.containsKey(negativeVertex))
                            headContextOfNegativeVertex = headContext.get(negativeVertex);
                        if (tailContext.containsKey(negativeVertex))
                            tailContextOfNegativeVertex = tailContext.get(negativeVertex);
                    }

                    int contextSize = 0;
                    if (headContextOfTrainingVertex != null)
                        contextSize += headContextOfTrainingVertex.size();
                    if (tailContextOfTrainingVertex != null)
                        contextSize += tailContextOfTrainingVertex.size();
                    double costTrainingVertex = (calcHeadContext(trainingVertex, headContextOfTrainingVertex) + calcTailContext(trainingVertex, tailContextOfTrainingVertex))/contextSize;
                    double costNegativeVertex = (calcHeadContext(negativeVertex, headContextOfTrainingVertex) + calcTailContext(negativeVertex, tailContextOfTrainingVertex))/contextSize;
                    if (costTrainingVertex - costNegativeVertex + margin > 0) {
                        loss += costTrainingVertex - costNegativeVertex + margin;
                        headGradient(trainingVertex, negativeVertex, headContextOfTrainingVertex);
                        tailGradient(trainingVertex, negativeVertex, tailContextOfTrainingVertex);
                    }

                    normalize(tmpEntityVec[trainingVertex]);
                    normalize(tmpEntityVec[negativeVertex]);
                    if (headContextOfTrainingVertex != null){
                        for (int i = 0; i < headContextOfTrainingVertex.size(); i++){
                            normalize(tmpRelationVec[headContextOfTrainingVertex.get(i)[0]]);
                            normalize(tmpEntityVec[headContextOfTrainingVertex.get(i)[1]]);
                        }
                    }
                    if (tailContextOfTrainingVertex != null){
                        for (int i = 0; i < tailContextOfTrainingVertex.size(); i++){
                            normalize(tmpRelationVec[tailContextOfTrainingVertex.get(i)[0]]);
                            normalize(tmpEntityVec[tailContextOfTrainingVertex.get(i)[1]]);
                        }
                    }
                }

                for (int i = 0; i < entityNum; i++)
                    for (int j = 0; j < n; j++)
                        entityVec[i][j] = tmpEntityVec[i][j];
                for (int i = 0; i < relationNum; i++)
                    for (int j = 0; j < n; j++)
                        relationVec[i][j] = tmpRelationVec[i][j];
            }

            System.out.println("epoch: " + epoch + ", loss: " + loss);

            outputCount++;
            if (outputCount == outputFreq){
                outputCount = 0;
                outputVectors(entityNum, relationNum);
            }
        }
    }

    public static void outputVectors(int entityNum, int relationNum) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter entityVecWrite = new PrintWriter(outputAddress + "\\entity2vector.txt", "UTF-8");
        PrintWriter relationVecWriter = new PrintWriter(outputAddress + "\\relation2vector.txt", "UTF-8");
        for (int i = 0; i < entityNum; i++){
            for (int j = 0; j < n; j++)
                entityVecWrite.printf("%.6f\t", entityVec[i][j]);
            entityVecWrite.print("\n");
        }
        for (int i = 0; i < relationNum; i++){
            for (int j = 0; j < n; j++)
                relationVecWriter.printf("%.6f\t", relationVec[i][j]);
            relationVecWriter.print("\n");
        }
        entityVecWrite.close();
        relationVecWriter.close();
    }

    public static void headGradient(int trainingVertex, int negativeVertex, ArrayList<int[]> headContext){
        if (headContext == null)
            return;
        for (int i = 0; i < headContext.size(); i++){
            for (int j = 0; j < n; j++){
                double x = entityVec[trainingVertex][j] + relationVec[headContext.get(i)[0]][j] - entityVec[headContext.get(i)[1]][j];
                if (x > 0){
                    tmpEntityVec[trainingVertex][j] -= rate;
                    tmpRelationVec[headContext.get(i)[0]][j] -= rate;
                    tmpEntityVec[headContext.get(i)[1]][j] += rate;
                }
                if (x < 0){
                    tmpEntityVec[trainingVertex][j] += rate;
                    tmpRelationVec[headContext.get(i)[0]][j] += rate;
                    tmpEntityVec[headContext.get(i)[1]][j] -= rate;
                }
            }
        }

        for (int i = 0; i <headContext.size(); i++){
            for (int j = 0; j < n; j++){
                double x = entityVec[negativeVertex][j] + relationVec[headContext.get(i)[0]][j] - entityVec[headContext.get(i)[1]][j];
                if (x > 0){
                    tmpEntityVec[negativeVertex][j] += rate;
                    tmpRelationVec[headContext.get(i)[0]][j] += rate;
                    tmpEntityVec[headContext.get(i)[1]][j] -= rate;
                }
                else {
                    tmpEntityVec[negativeVertex][j] -= rate;
                    tmpRelationVec[headContext.get(i)[0]][j] -= rate;
                    tmpEntityVec[headContext.get(i)[1]][j] += rate;
                }
            }
        }
    }

    public static void tailGradient(int trainingVertex, int negativeVertex, ArrayList<int[]> tailContext){
        if (tailContext == null)
            return;
        for (int i = 0; i < tailContext.size(); i++){
            for (int j = 0; j < n; j++){
                double x = entityVec[tailContext.get(i)[1]][j] + relationVec[tailContext.get(i)[0]][j] - entityVec[trainingVertex][j];
                if (x > 0){
                    tmpEntityVec[tailContext.get(i)[1]][j] -= rate;
                    tmpRelationVec[tailContext.get(i)[0]][j] -= rate;
                    tmpEntityVec[trainingVertex][j] += rate;
                }
                if (x < 0){
                    tmpEntityVec[tailContext.get(i)[1]][j] += rate;
                    tmpRelationVec[tailContext.get(i)[0]][j] += rate;
                    tmpEntityVec[trainingVertex][j] -= rate;
                }
            }
        }

        for (int i = 0; i < tailContext.size(); i++){
            for (int j = 0; j < n; j++){
                double x = entityVec[tailContext.get(i)[1]][j] + relationVec[tailContext.get(i)[0]][j] - entityVec[negativeVertex][j];
                if (x > 0){
                    tmpEntityVec[tailContext.get(i)[1]][j] += rate;
                    tmpRelationVec[tailContext.get(i)[0]][j] += rate;
                    tmpEntityVec[negativeVertex][j] -= rate;
                }
                else {
                    tmpEntityVec[tailContext.get(i)[1]][j] -= rate;
                    tmpRelationVec[tailContext.get(i)[0]][j] -= rate;
                    tmpEntityVec[negativeVertex][j] += rate;
                }
            }
        }
    }

    public static double calcHeadContext(int vertex, ArrayList<int[]> headContext){
        if (headContext == null)
            return 0;
        double cost = 0;
        for (int j = 0; j < headContext.size(); j++) {
            double tmpCost = 0;
            for (int i = 0; i < n; i++)
                tmpCost += Math.pow(entityVec[vertex][i] + relationVec[headContext.get(j)[0]][i] - entityVec[headContext.get(j)[1]][i], 2);
            tmpCost = Math.sqrt(tmpCost);
            cost += tmpCost;
        }
        return cost;
    }

    public static double calcTailContext(int vertex, ArrayList<int[]> tailContext){
        if (tailContext == null)
            return 0;
        double cost = 0;
        for (int j =0; j < tailContext.size(); j++){
            double tmpCost = 0;
            for (int i = 0; i < n; i++)
                tmpCost += Math.pow(entityVec[tailContext.get(j)[1]][i] + relationVec[tailContext.get(j)[0]][i] - entityVec[vertex][i], 2);
            tmpCost = Math.sqrt(tmpCost);
            cost += tmpCost;
        }
        return cost;
    }

    public static boolean arrayListCompare(ArrayList<int[]> arr1, ArrayList<int[]> arr2){  //return true if arr1 and arr2 do not intersect
        if (arr1 == null || arr2 == null)
            return true;
        for (int[] anArr1 : arr1) {
            if (arr2.contains(anArr1))
                return false;
        }
        return true;
    }

    public void initVectors(boolean preOrRaw, String preVectorsAddress, int entityNum, int relationNum) throws IOException {
        System.out.println("--------------------------------------------------------");
        if (!preOrRaw){
            System.out.println("initializing embedding vectors...");
            initializingVectors(entityNum, relationNum);
        }
        else {
            System.out.println("loading pre-trained entity vectors...");
            readEntity2Vec(preVectorsAddress, entityNum);
            System.out.println("loading pre-trained relation vectors...");
            readRelation2Vec(preVectorsAddress, relationNum);
        }

    }

    public void initializingVectors(int entityNum, int relationNum){
        entityVec = new double[entityNum][n];
        relationVec = new double[relationNum][n];

        for (int i = 0; i < entityNum; i++) {
            for (int j = 0; j < n; j++) {
                entityVec[i][j] = (2 * Math.random() - 1.0) * 6 / Math.sqrt(n);
            }
            normalize(entityVec[i]);
        }

        for (int i = 0; i < relationNum; i++) {
            for (int j = 0; j < n; j++) {
                relationVec[i][j] = (2 * Math.random() - 1.0) * 6 / Math.sqrt(n);
            }
            normalize(relationVec[i]);
        }
        if (entityVec.length == entityNum && relationVec.length == relationNum)
            System.out.println("initializing complete.");
        else
            System.out.println("initializing failed.");

    }

    public void readEntity2Vec(String preVectorsAddress, int entityNum) throws IOException {
        entityVec = new double[entityNum][n];
        preVectorsAddress += "\\entity2vector.txt";
        BufferedReader reader = new BufferedReader(new FileReader(new File(preVectorsAddress)));
        String line;
        int i = 0;
        while((line = reader.readLine()) != null){
            line = line.trim();
            String[] tmpVector = line.split("\t");
            if(tmpVector.length != n){
                System.out.println("the dimension of pre-learned entity vectors is not " + n +"!");
                System.exit(1);
            }
            for(int j=0; j<n; j++)
                entityVec[i][j] = Double.valueOf(tmpVector[j]);
            i++;
        }
        reader.close();
        if (i == entityNum)
            System.out.println("loading of pre-learned entity vectors complete.");
        else
            System.out.println("the number of pre-learned entity vectors is " + i + " instead of " + entityNum +"!");
    }

    public void readRelation2Vec(String preVectorsAddress, int relationNum) throws IOException {
        relationVec = new double[relationNum][n];
        preVectorsAddress += "\\relation2vector.txt";
        BufferedReader reader = new BufferedReader(new FileReader(new File(preVectorsAddress)));
        String line;
        int i = 0;
        while ((line = reader.readLine()) != null){
            line = line.trim();
            String[] tmpVector = line.split("\t");
            if (tmpVector.length != n){
                System.out.println("the dimension of pre-learned relation vectors is not " + n +"!");
                System.exit(1);
            }
            for (int j=0; j<n; j++)
                relationVec[i][j] = Double.valueOf(tmpVector[j]);
            i++;
        }
        reader.close();
        if (i == relationNum)
            System.out.println("loading of pre-learned relation vectors complete.");
        else
            System.out.println("the number of pre-learned relation vectors is " + i + " instead of " + relationNum +"!");
    }

    public static void normalize(double[] vec) {
        double sum = 0;
        for (int i = 0; i < vec.length; i++) {
            double aVec = vec[i];
            sum += Math.pow(aVec, 2);
        }
        for (int i = 0; i < vec.length; i++) {
            vec[i] /= Math.sqrt(sum);
        }
    }

    public void beginningStatement(){
        System.out.println("--------------------------------------------------------");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        System.out.println("launching time: " + df.format(new Date()));
        System.out.println("the address of input data: " + inputAddress);
        if(!preOrRaw)
            System.out.println("learn from the beginning");
        else
        {
            System.out.println("learn based on pre-learned vectors");
            System.out.println("the address of pre-learned vectors: " +  preVectorsAddress);
        }
        System.out.println("dimension of learned vectors: " + n);
        System.out.println("number of epoch: " + epochNum);
        System.out.println("learning rate: " + rate);
        System.out.println("margin: " + margin);
        System.out.println("number of batches: " + batchNum);
        System.out.println("output address: " + outputAddress);
    }

    public void enddingStatement(){
        System.out.println("--------------------------------------------------------");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        System.out.println("exiting time: " + df.format(new Date()));
        System.out.println("--------------------------------------------------------");
    }

}
