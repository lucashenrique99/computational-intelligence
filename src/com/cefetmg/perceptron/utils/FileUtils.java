package com.cefetmg.perceptron.utils;

import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

public class FileUtils {
	
	public static String [] getFileList_FromPath(String filePath){
		File f = new File(filePath);
		String [] fileArr = null;
		if (f.isDirectory()){
			fileArr = f.list();
		}

		if (fileArr ==null)
			fileArr = new String [0];
		
		return fileArr; 
    }
	
	public static boolean existFile(String filePath, String fileName){
		File f = new File(filePath + fileName);
		return f.exists();
	}

	public static void move(File f, String filePathNew){
		String fileNameNew = f.getName();
		move(f, filePathNew, fileNameNew);
	}
	
	public static void move(File f, String filePathNew, String fileNameNew){
		String filePath = f.getParent() + "/";
		String fileName = f.getName(); 
		move(filePath, filePathNew, fileName, fileNameNew);
	}
	
	public static void move(String filePath, String filePathNew, String fileName){
		move(filePath, filePathNew, fileName, fileName);
	}
	
	public static void move(String filePath, String filePathNew, String fileName, String fileNameNew){
		File f = new File(filePath + fileName);
		
		File f1 = new File(filePathNew + fileNameNew);
		File dir = new File(filePathNew);
		
		if ( (f.exists()) && (dir.exists()) ){
			f.renameTo(f1);
		}
	}
	
	public static Object openObject(String filePath, String fileName){
		Object objRet = null;
		File f = new File(filePath + fileName);
		
		ObjectInputStream objIn = null;
    	try {
    		objIn = new ObjectInputStream(new FileInputStream(f));
    		objRet = objIn.readObject();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}finally{
			try {
				objIn.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		return objRet;
	}
	
	public static List<Object> openObjectList(String filePath, String fileName){
		List<Object> objListRet = new ArrayList<Object>();
		File f = new File(filePath + fileName);
		
		if (f.exists()){
			ObjectInputStream objIn = null;
	    	try {
	    		objIn = new ObjectInputStream(new FileInputStream(f));
	    		Object obj = objIn.readObject();
	    		while(obj != null){
	    			objListRet.add(obj);
	    			obj = objIn.readObject();	
	    		}
	    		
			} catch (EOFException e) {
				//e.printStackTrace();
				System.out.println(objListRet.size() + " objeto(s) lido(s)");
				System.out.println("Fim de arquivo: " +f.getPath());
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			} catch (Throwable e) {
				e.printStackTrace();
			} finally{
				try {
					objIn.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return objListRet;
	}
	
	public static void saveObject(String filePath, String fileName, Object obj){
		File f = new File(filePath + fileName);
		
		ObjectOutputStream objOut = null;
    	try {
			objOut = new ObjectOutputStream(new FileOutputStream(f));
			objOut.writeObject(obj);

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}finally{
			try {
				objOut.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
    }
	
	public static void saveObject(String filePath, String fileName, Object obj, int qtdVersions){
		if (qtdVersions < 0)
			return;
			
		for(int i = qtdVersions; i > 0; i--){
			//Se o indice anterior do arquivo for 0 entao nao se usa numero para a versao.
			String idxAnt = "" + (i-1);
			if (i == 1)
				idxAnt = "";
			
			
			File f = new File(filePath + fileName + idxAnt);
			File f1 = new File(filePath + fileName + i);
			if (f.exists()){
				if (i == qtdVersions){
					f1.delete();
				} else{
					if (f1.exists()){
						f1.delete();
					}
					f.renameTo(f1);
				}
				
			}	
		}
		
		saveObject(filePath, fileName, obj);
    }
	
	public static void saveObjectList(String filePath, String fileName, List<Object> objList){
		File f = new File(filePath + fileName);
		
		ObjectOutputStream objOut = null;
    	try {
			objOut = new ObjectOutputStream(new FileOutputStream(f));
			for (Object obj : objList) {
				objOut.writeObject(obj);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}finally{
			try {
				objOut.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
    }

	
	
}
