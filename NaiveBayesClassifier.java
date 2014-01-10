import java.io.*;

/**
 * A Naive Bayes Classifier.
 */
public class NaiveBayesClassifier implements Classifier {

	//the number of label values is always 2
	private final int NUMBER_OF_LABEL_VALUES = 2;
		
	//index[l][F_i][f_i] is the log probability that feature F_i
	//takes the value attrVals[F_i][f_i] given that the label takes the value l
	private double[][][] logProbabilityAttributeTakesValueIndexedByLabelValue;
	
	//index[l] is the log probability that the label takes the value l
	private double[] logProbabilityLabelTakesValue;
	
    /**
     * This constructor takes as input a discrete data set.
     */
    public NaiveBayesClassifier(DiscreteDataSet d) {
		
		//initialize array logProbabilityLabelTakesValue 
		logProbabilityLabelTakesValue = new double[NUMBER_OF_LABEL_VALUES];
		
		//initialize array logProbabilityAttributeTakesValueIndexedByLabelValue
		logProbabilityAttributeTakesValueIndexedByLabelValue 
			= new double[NUMBER_OF_LABEL_VALUES][d.numAttrs][];
		
		//get the maximum attribute value length for the sake
		//of the numberTimesAttributeTakesValueIndexedByLabelValue array we
		//are about to create 
		int maxAttributeValueLength = -1;
		for (int i = 0; i < d.attrVals.length; i++)
		{
			if (d.attrVals[i].length > maxAttributeValueLength)
			{
				maxAttributeValueLength = d.attrVals[i].length;
			}
		}
		
		//index [l][F_i][f_i] is the number of times feature F_i takes
		//the value f_i when the label takes the value l
		int[][][] numberTimesAttributeTakesValueIndexedByLabelValue
			= new int[NUMBER_OF_LABEL_VALUES][d.numAttrs][maxAttributeValueLength];
		
		//index [l] is the number of times in the training data the label takes
		//the value l
		int[] numberOfTimesLabelTakesValue = new int[NUMBER_OF_LABEL_VALUES];
		
		//calculate number of times each attribute takes each value it takes
		//based on the label value in the training data
		for (int i = 0; i < d.numTrainExs; i++)
		{
			numberOfTimesLabelTakesValue[d.trainLabel[i]]++;
			for (int j = 0; j < d.trainEx[i].length; j++)
			{
				numberTimesAttributeTakesValueIndexedByLabelValue[d.trainLabel[i]][j][d.trainEx[i][j]]++;
			}
		}
		
		//calculate the log probability the label takes each value it takes
		for (int i = 0; i < logProbabilityLabelTakesValue.length; i++)
		{
			logProbabilityLabelTakesValue[i] = Math.log((1 + numberOfTimesLabelTakesValue[i])
				/ (double) (NUMBER_OF_LABEL_VALUES + d.numTrainExs));
		}
		
		//calculate the log probability each feature takes each value it takes based
		//on the label value
		for (int i = 0; i < NUMBER_OF_LABEL_VALUES; i++)
		{
			for (int j = 0; j < d.numAttrs; j++)
			{
				logProbabilityAttributeTakesValueIndexedByLabelValue[i][j] 
					= new double[d.attrVals[j].length];
				for (int k = 0; k < logProbabilityAttributeTakesValueIndexedByLabelValue[i][j].length;
					k++)
				{
					logProbabilityAttributeTakesValueIndexedByLabelValue[i][j][k]
						= Math.log((1 + numberTimesAttributeTakesValueIndexedByLabelValue[i][j][k])
							/ (double)(logProbabilityAttributeTakesValueIndexedByLabelValue[i][j].length
							+ d.numTrainExs)) - logProbabilityLabelTakesValue[i];
				}
			}
		}
    }

	/**
	 * Tests the class. Assumes the existence of a dataset
	 * with the stem <gender> in the same directory. 
	 */
	public static void testConstructor() throws FileNotFoundException, IOException 
	{
		DiscreteDataSet d = new DiscreteDataSet("gender");

		NaiveBayesClassifier c = new NaiveBayesClassifier(d);
		
		double[] logProbabilityLabelTakesValue = c.getLogProbabilityLabelTakesValue();
		double[][][] logProbabilityAttributeTakesValueIndexedByLabelValue
			= c.getLogProbabilityAttributeTakesValueIndexedByLabelValue();
		
		NaiveBayesClassifier.assertDoubleEquals(logProbabilityLabelTakesValue[0],
			-0.6931471805599453);
		
		NaiveBayesClassifier.assertDoubleEquals(logProbabilityLabelTakesValue[0],
			-0.6931471805599453);
			
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[0][0][0],
			-0.5596157879354228);
		
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[0][0][1],
			-1.252762968495368);
		
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[0][0][2],
			-0.5596157879354228);
			
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[0][1][0],
			-0.15415067982725839);
			
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[0][1][1],
			-1.252762968495368);
			
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[0][1][2],
			-1.252762968495368);
			
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[1][0][0],
			-0.5596157879354228);
		
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[1][0][1],
			-0.5596157879354228);
		
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[1][0][2],
			-1.252762968495368);
			
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[1][1][0],
			-1.252762968495368);
			
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[1][1][1],
			-0.5596157879354228);
			
		NaiveBayesClassifier.assertDoubleEquals(
			logProbabilityAttributeTakesValueIndexedByLabelValue[1][1][2],
			-0.5596157879354228);
			
		//test the prediction on an example of "tall average" output should be woman
		
		int label = c.predict(new int[]{0, 1});
		
		NaiveBayesClassifier.assertDoubleEquals((double) label, 1.0);
	}
	
	/**
	 * Returns the logProbabilityLabelTakesValue array.
	 */
	public double[] getLogProbabilityLabelTakesValue() {
		return logProbabilityLabelTakesValue;
	}
	
	/**
	 * Returns the logProbabilityLabelTakesValue array.
	 */
	public double[][][] getLogProbabilityAttributeTakesValueIndexedByLabelValue() {
		return logProbabilityAttributeTakesValueIndexedByLabelValue;
	}
	
	/**
	 * Exits with an error if a is not equal to b.
	 */
	public static void assertDoubleEquals(double a, double b)
	{
		if (a != b)
		{
			System.out.println("Error: " + a + " is not equal to " + b);
			System.exit(1);
		}
	}
	
    /** 
     * Predictes the label of ex. 
     */
    public int predict(int[] ex) {
		
		double maxLogSum = -Double.MAX_VALUE;
		int maximizingLabel = -1;
		
		for (int i = 0; i < NUMBER_OF_LABEL_VALUES; i++)
		{
			double logSum = logProbabilityLabelTakesValue[i];
			for (int j = 0; j < ex.length; j++)
			{
				logSum += logProbabilityAttributeTakesValueIndexedByLabelValue[i][j][ex[j]];
			}
			
			if (logSum > maxLogSum)
			{
				maxLogSum = logSum;
				maximizingLabel = i;
			}
		}
		
		return maximizingLabel;
    }

    /** This method returns a description of the learning algorithm. */
    public String algorithmDescription() {
		return "Implements Naive Bayes using discretization as opposed to continuous values.";
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

		DiscreteDataSet d = new DiscreteDataSet(filestem);

		Classifier c = new NaiveBayesClassifier(d);

		d.printTestPredictions(c, filestem);
    }

}
