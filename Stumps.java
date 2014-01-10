/**
 * @(#)Stumps.java
 *
 *
 * @author
 * @version 1.00 2014/1/8
 */
import java.util.*;
import java.io.*;
public class Stumps implements Classifier {
	private boolean bool_decision = true;
	private int index_decision = 0;
    public Stumps(BinaryDataSet b, double[] weights) {
    	//for thru attributes:
    		//check if this attribute is the one we should split on by
    		//for thru training examples
    			//split on this attribute, see how many (weighted) mistakes
    	double max_decision = 0;
    	for (int a = 0; a < b.numAttrs; a++) {
	    	double match = 0;
	    	double mistake = 0;
			for (int ex = 0; ex < b.numTrainExs; ex++) {
				if (b.trainEx[ex][a] == b.trainLabel[ex])
					match += weights[ex];
				else
					mistake += weights[ex];
			}
			if (match > max_decision) {
				bool_decision = true;
				index_decision = a;
				max_decision = match;
			}
			if (mistake > max_decision) {
				bool_decision = false;
				index_decision = a;
				max_decision = mistake;
			}
    	}
    }

    public int predict(int[] ex) {
		if (ex[index_decision] == 0) {
			if (bool_decision)
				return 0;
			else
				return 1;
		}
		else {
			if (bool_decision)
				return 1;
			else
				return 0;
		}
    }
    public String algorithmDescription() {
    	return "weighted stump";
    }
	public String author() {
		return "pthorpedmmckenn";
	}
	public static void main(String[] args)

	throws FileNotFoundException, IOException {

		if (args.length < 1) {
		    System.err.println("argument: filestem");
		    return;
		}

		String filestem = args[0];

		BinaryDataSet d = new BinaryDataSet(filestem);
		double[] weights = new double[d.numTrainExs];
		for (int i = 0; i < weights.length; i++)
			weights[i] = 1;
		Classifier c = new Stumps(d,weights);

		d.printTestPredictions(c, filestem);
	}

}
