package model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.log4j.BasicConfigurator;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.structure.Graph;
import org.apache.tinkerpop.gremlin.structure.T;
import org.apache.tinkerpop.gremlin.structure.Vertex;
import org.apache.tinkerpop.gremlin.tinkergraph.structure.TinkerGraph;
import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;

public class createNeighborMap {
	
	Map<String, Integer> entity2id = new HashMap<>();
	Map<String, Integer> relation2id = new HashMap<>();
	List<Triple> trainingDataObjSet = new ArrayList<>();
	public int entityNum = 0;
	
	public createNeighborMap(String filePath) throws IOException {
		super();
		BasicConfigurator.configure();
		readEntityIdFromFile("entity2id.txt");
		readRelationIdFromFile("relation2id.txt");
		NeighborContext n = new NeighborContext();
		n = readTrainingData(filePath);
		Gson gson = new Gson();
		FileWriter writer = new FileWriter("Neighbor.json");
		gson.toJson(n , writer);
		writer.close();
		System.out.println("Done");
	}
	
	void readEntityIdFromFile(String filePath) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
		String line = "";
		while ((line = reader.readLine()) != null) {
			String[] array = line.split(" ");
			String entityName = array[0].trim();
			int entityId = Integer.valueOf(array[1].trim());
			entity2id.put(entityName, entityId);
			entityNum++;
		}
		reader.close();
	}
	
	void readRelationIdFromFile(String filePath) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(new File(filePath)));
		String line = "";
		while ((line = reader.readLine()) != null) {
			String[] array = line.split(" ");
			String relationName = array[0].trim();
			int relationId = Integer.valueOf(array[1].trim());
			relation2id.put(relationName, relationId);
		}
		reader.close();
	}
	
	
	
	public NeighborContext readTrainingData(String filePath) {

		//存放内容的map对象
		NeighborContext Neighbor = new NeighborContext();
		String line = "";
		try {
		String encoding = "GBK";
		File file = new File(filePath);
		if (file.isFile() && file.exists()) { 
		InputStreamReader read = new InputStreamReader(
		new FileInputStream(file), encoding);
		BufferedReader reader = new BufferedReader(read);
		while ((line = reader.readLine()) != null) {
			String[] triple = line.split(" ");
			String head = triple[0].trim();
			String relation = triple[1].trim();
			String tail = triple[2].trim();
			int headId = entity2id.get(head);
			int relationId = relation2id.get(relation);
			int tailId = entity2id.get(tail);
			trainingDataObjSet.add(new Triple(headId, relationId, tailId));
		}
		reader.close();
		for(int i=0; i<entityNum; i++) {
			Set<Neighbor> set = new HashSet<>();
			System.out.println(i);
			for(Triple t : trainingDataObjSet){
				if(t.head == i) {
					set.add(new Neighbor(t.relation, t.tail));
				}
				if(t.tail == i) {
					set.add(new Neighbor(t.relation, t.head));
				}
			}
			Neighbor.put(i, set);
		}
		
		}
		 else {
		System.out.println("找不到指定的文件");
		}
		} catch (Exception e) {
		System.out.println("读取文件内容出错");
		e.printStackTrace();
		}
		return Neighbor;
	}
	
	public static void main(String[] args) throws IOException{
		createNeighborMap c = new createNeighborMap("train.txt");
	}
}
