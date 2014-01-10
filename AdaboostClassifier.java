import java.io.*;

/**
 * Implements Adaboost with decision stumps as the weak learner.
 */
public class AdaboostClassifier implements Classifier {

	//the hypotheses in the ensemble
	private Stumps[] h;
    
    //the hypothesis weights
    private double[] z;
    
    /**
     * This constructor takes as input a binary dataset and a number k
     * which specifies the number of stump learners in the ensemble.
     */
    public AdaboostClassifier(BinaryDataSet d, int k) {
		
		//initialize the hypotheses vector and the hypothesis weights vector
		h = new Stumps[k];
		z = new double[k];
		
		//initialize the example weights vector to initially be 1/n for all examples
		double[] w = new double[d.numTrainExs];
		for (int i = 0; i < w.length; i++) w[i] = 1 / (double) d.numTrainExs;
		
		for (int i = 0; i < k; i++)
		{
			h[i] = new Stumps(d, w);
			
			double error = 0;
			
			for (int j = 0; j < d.numTrainExs; j++)
			{
				int prediction = h[i].predict(d.trainEx[j]);
				
				if (prediction != d.trainLabel[j])
				{
					error += w[j];
				}
			}
			
			for (int j = 0; j < d.numTrainExs; j++)
			{
				int prediction = h[i].predict(d.trainEx[j]);
				
				if (prediction == d.trainLabel[j])
				{
					w[j] *= (error) / (1 - error);
				}
			}
			
			//normalize w
			double wSum = 0.0;
			for (int j = 0; j < w.length; j++) wSum += w[j];
			for (int j = 0; j < w.length; j++) w[j] /= wSum;
			
			z[i] = Math.log((1 - error) / error);
		}
    }

    /** 
     * Predicts Weighted-Majority(h,z).
	 * Weighted-Majority generates a hypothesis that returns the output value with the
	 * highest vote from the hypotheses in h, with votes weighted by z.
     */
    public int predict(int[] ex) {
		
		//index [i] is the weighted votes sum for the label taking the value i
		double[] labelVotes = new double[2];
		
		//iterate over all hypotheses, counting their votes, weighted by z
		for (int i = 0; i < h.length; i++)
		{
			int prediction = h[i].predict(ex);
			labelVotes[prediction] += z[i];
		}
		
		return labelVotes[0] > labelVotes[1] ? 0 : 1;
    }

    /** This method returns a description of the learning algorithm. */
    public String algorithmDescription() {
		return "Implements Adaboost with 1,000 rounds of boosting with decision stumps as the weak learner.";
    }

    /** This method returns the author of this program. */
    public String author() {
		return "dmmckenn_pthorpe";
    }

    /** A simple main for testing this algorithm.  This main reads a
     * filestem from the command line, runs the learning algorithm on
     * this dataset, and prints the test predictions to filestem.testout.
     */
    public static void main(String argv[])
	throws FileNotFoundException, IOException {

		if (argv.length < 1) {
			System.err.println("argument: filestem");
			return;
		}

		String filestem = argv[0];

		BinaryDataSet d = new BinaryDataSet(filestem);

		Classifier c = new AdaboostClassifier(d, 1000);

		d.printTestPredictions(c, filestem);
    }

}
