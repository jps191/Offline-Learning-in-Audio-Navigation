using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class ListeningAgent : Agent
{
	public GameObject speakerA;
	public GameObject speakerB;
	public GameObject speakerC;
	public GameObject targetSpeaker;
	public bool isTraining = false;
	public int maxSpeakerClips = 0;
	public int numSamples = 1024;
	public float forceMultiplier = 10;
	public float minimumDistance = 4.0f;
    public bool humanControl = false;
    private Rigidbody rBody;
    private Object[] speakerAClips;
    private Object[] speakerBClips;
    private Object[] speakerCClips;
    private AudioSource speakerAAudioSource;
    private AudioSource speakerBAudioSource;
    private AudioSource speakerCAudioSource;
    private Renderer targetSpeakerRenderer;
    private Renderer targetSpeakerChildRenderer;
    
    private int stepCounter = 0;
    private float totalReward = 0;
    
    // Success rate tracking
    private static int totalEpisodes = 0;
    private static int successfulEpisodes = 0;
    private static float cumulativeReward = 0;
    private const int LOG_INTERVAL = 100;
    
    // Start is called before the first frame update
    void Start() {
		rBody = GetComponent<Rigidbody>(); 
		
		// Load audio clips for each speaker
		if (isTraining) {
			speakerAClips = Resources.LoadAll("Audio/SpeakerA/Train");
			Debug.Log("Loaded train audio clips for speaker A.");
			speakerBClips = Resources.LoadAll("Audio/SpeakerB/Train");
			Debug.Log("Loaded train audio clips for speaker B.");
			speakerCClips = Resources.LoadAll("Audio/SpeakerC/Train");
			Debug.Log("Loaded train audio clips for speaker C.");
		} else {
			speakerAClips = Resources.LoadAll("Audio/SpeakerA/Test");
			Debug.Log("Loaded test audio clips for speaker A.");
			speakerBClips = Resources.LoadAll("Audio/SpeakerB/Test");
			Debug.Log("Loaded test audio clips for speaker B.");
			speakerCClips = Resources.LoadAll("Audio/SpeakerC/Test");
			Debug.Log("Loaded test audio clips for speaker C.");	
		}
		
		// Limit number of audio clips per speaker
		if (maxSpeakerClips > 0) {
			Object[] tempA = new Object[maxSpeakerClips];
			Object[] tempB = new Object[maxSpeakerClips];
			Object[] tempC = new Object[maxSpeakerClips];
			for (int i = 0; i < maxSpeakerClips; ++i) {
				tempA[i] = speakerAClips[i];
				tempB[i] = speakerBClips[i];
				tempC[i] = speakerCClips[i];
			}
			speakerAClips = tempA;
			speakerBClips = tempB;
			speakerCClips = tempC;
		}
		
		// Get AudioSources
		speakerAAudioSource = speakerA.GetComponent<AudioSource>();
		speakerBAudioSource = speakerB.GetComponent<AudioSource>();
		speakerCAudioSource = speakerC.GetComponent<AudioSource>();
		
		// Set target speaker color
		targetSpeakerRenderer = targetSpeaker.GetComponent<Renderer>();
		targetSpeakerRenderer.material.color = Color.green;
		targetSpeakerChildRenderer = targetSpeaker.transform.GetChild(0).gameObject.GetComponent<Renderer>();
		targetSpeakerChildRenderer.material.color = Color.green;
    }
	
	public void InitAgent() {
		// Spawn agent in a random location with zero momentum
		this.rBody.angularVelocity = Vector3.zero;
        this.rBody.velocity = Vector3.zero;
		this.transform.localPosition = new Vector3(Random.Range(-4.0f, 4.0f),
												   0.5f,
												   4.0f);
	}
	
	public void InitSpeakers() {
		// Spawn speakers in random locations
		speakerA.transform.localPosition = new Vector3(Random.Range(-4.0f, 4.0f),
													   1.5f,
											           Random.Range(-4.0f, 4.0f));
		speakerB.transform.localPosition = new Vector3(Random.Range(-4.0f, 4.0f),
											           2.0f,
											           Random.Range(-4.0f, 4.0f));
		speakerC.transform.localPosition = new Vector3(Random.Range(-4.0f, 4.0f),
											           1.75f,
											           Random.Range(-4.0f, 4.0f));
	}
	
	public override void OnEpisodeBegin() {
		// Spawn agent in a random location with zero momentum
		InitAgent();
		// Spawn speakers in random locations
		InitSpeakers();
		// Repeat until speakers are further apart than a minimum distance
		float distanceAtoB = Vector3.Distance(speakerA.transform.localPosition, speakerB.transform.localPosition);
		float distanceAtoC = Vector3.Distance(speakerA.transform.localPosition, speakerC.transform.localPosition);
		float distanceBtoC = Vector3.Distance(speakerB.transform.localPosition, speakerC.transform.localPosition);
		float distanceAtoAgent = Vector3.Distance(speakerA.transform.localPosition, this.transform.localPosition);
		float distanceBtoAgent = Vector3.Distance(speakerB.transform.localPosition, this.transform.localPosition);
		float distanceCtoAgent = Vector3.Distance(speakerC.transform.localPosition, this.transform.localPosition);
		while (distanceAtoB < minimumDistance || distanceAtoC < minimumDistance || distanceBtoC < minimumDistance || distanceAtoAgent < minimumDistance || distanceBtoAgent < minimumDistance || distanceCtoAgent < minimumDistance) {
			InitSpeakers();
			distanceAtoB = Vector3.Distance(speakerA.transform.localPosition, speakerB.transform.localPosition);
			distanceAtoC = Vector3.Distance(speakerA.transform.localPosition, speakerC.transform.localPosition);
			distanceBtoC = Vector3.Distance(speakerB.transform.localPosition, speakerC.transform.localPosition);
			distanceAtoAgent = Vector3.Distance(speakerA.transform.localPosition, this.transform.localPosition);
			distanceBtoAgent = Vector3.Distance(speakerB.transform.localPosition, this.transform.localPosition);
			distanceCtoAgent = Vector3.Distance(speakerC.transform.localPosition, this.transform.localPosition);
		}    
	}
	
	public float[] GetSpeakerVector() {
		// Debug.Log("Target speaker is " + targetSpeaker.name);
		float[] speakerVector = {0.0f, 0.0f, 0.0f};
		if (targetSpeaker.name == "SpeakerA") {
			speakerVector[0] = 1.0f;
		} else if (targetSpeaker.name == "SpeakerB") {
			speakerVector[1] = 1.0f;
		} else if (targetSpeaker.name == "SpeakerC") {
			speakerVector[2] = 1.0f;
		} else {
			Debug.Log("Unknown speaker object " + targetSpeaker.name);
		}
		// Debug.Log(speakerVector);
		return speakerVector;
	}
	
	public override void CollectObservations(VectorSensor sensor) {
		// Construct audio data inputs
		float[] audioDataLeft = new float[numSamples];
		float[] audioDataRight = new float[numSamples];
		AudioListener.GetOutputData(audioDataLeft, 0);
		AudioListener.GetOutputData(audioDataRight, 1);
		float[] audioData = new float[2 * numSamples];
		for (int i = 0; i < numSamples; ++i) {
			audioData[i] = audioDataLeft[i];
		}
		for (int i = numSamples; i < 2 * numSamples; ++i) {
			audioData[i] = audioDataRight[i - numSamples];
		}
		// Contrust speaker vector inputs
		float[] speakerVector = GetSpeakerVector();
		// Add audio data observation
		sensor.AddObservation(audioData);
		// Add observation for target speaker
		// sensor.AddObservation(speakerVector);
		// TOTAL INPUT VECTOR SIZE = (numSamples * 2) + 3
		// e.g. 2048 + 3 = 2051
	}
	
	public override void OnActionReceived(ActionBuffers actions) {
		// Actions, size = 2
		Vector3 controlSignal = Vector3.zero;
		controlSignal.x = actions.ContinuousActions[0];
		controlSignal.z = actions.ContinuousActions[1];
		rBody.AddForce(controlSignal * forceMultiplier);
		
		// Rewards
		float distanceToTarget = Vector3.Distance(this.transform.localPosition, targetSpeaker.transform.GetChild(0).position);
		// Reached target
		if (distanceToTarget < 1.4f) {
			SetReward(1.0f);
			Debug.Log("Reward 1.0 for reaching target speaker.");
			totalReward += 1.0f;
			Debug.Log("Completed episode in " + stepCounter + " steps.");
			Debug.Log("Total reward = " + totalReward);
			
			// Track success
			successfulEpisodes++;
			totalEpisodes++;
			cumulativeReward += totalReward;
			LogSuccessRate();
			
			EndEpisode();
			stepCounter = 0;
			totalReward = 0;
		}
		GameObject[] speakers = {speakerA, speakerB, speakerC};
		for (int i = 0; i < speakers.Length; ++i) {
			float distanceToSpeaker = Vector3.Distance(this.transform.localPosition, speakers[i].transform.GetChild(0).position);
			if (distanceToSpeaker < 1.4f) {
				// Reached another speaker that is not the target
				SetReward(-1.0f);
				Debug.Log("Reward -1.0 for reaching wrong speaker.");
				totalReward += -1.0f;
				Debug.Log("Completed episode in " + stepCounter + " steps.");
				Debug.Log("Total reward = " + totalReward);
				
				// Track failure
				totalEpisodes++;
				cumulativeReward += totalReward;
				LogSuccessRate();
				
				EndEpisode();
				stepCounter = 0;
				totalReward = 0;
			}
		}
		// Fell off platform
		if (this.transform.localPosition.y < 0.45f) {
			SetReward(-1.0f);
			Debug.Log("Reward -1.0 for falling off platform.");
			totalReward += -1.0f;
			Debug.Log("Completed episode in " + stepCounter + " steps.");
			Debug.Log("Total reward = " + totalReward);
			
			// Track failure
			totalEpisodes++;
			cumulativeReward += totalReward;
			LogSuccessRate();
			
			EndEpisode();
			stepCounter = 0;
			totalReward = 0;
		}
		// Small negative reward for every step
		SetReward(-0.001f);
		++stepCounter;
		totalReward += -0.001f;
	}
	
	public override void Heuristic(in ActionBuffers actionsOut) {
		var continuousActionsOut = actionsOut.ContinuousActions;
		if (humanControl) {
			continuousActionsOut[0] = -Input.GetAxis("Horizontal");
			continuousActionsOut[1] = -Input.GetAxis("Vertical");
		} else {
			continuousActionsOut[0] = Random.Range(-1.0f, 1.0f);
			continuousActionsOut[1] = Random.Range(-1.0f, 1.0f);
		}
	}
	
	private void LogSuccessRate() {
		if (totalEpisodes % LOG_INTERVAL == 0) {
			float successRate = (float)successfulEpisodes / totalEpisodes * 100f;
			float averageReward = cumulativeReward / totalEpisodes;
			string report = "=== SUCCESS RATE REPORT ===\n" +
			                "Total Episodes: " + totalEpisodes + "\n" +
			                "Successful Episodes: " + successfulEpisodes + "\n" +
			                "Success Rate: " + successRate.ToString("F2") + "%\n" +
			                "Average Reward: " + averageReward.ToString("F4") + "\n" +
			                "===========================";
			Debug.Log(report);
			UnityEngine.Debug.LogFormat(LogType.Log, LogOption.NoStacktrace, null, report);
			System.Console.WriteLine(report);
		}
	}
	
    // Update is called once per frame
    void Update() {
		// If no clip is playing from speaker AudioSource
		if (!speakerAAudioSource.isPlaying) {
			// Assign a random clip and play it
			speakerAAudioSource.clip = (AudioClip)speakerAClips[Random.Range(0, speakerAClips.Length)];
			speakerAAudioSource.Play();
		}
		if (!speakerBAudioSource.isPlaying) {
			speakerBAudioSource.clip = (AudioClip)speakerBClips[Random.Range(0, speakerBClips.Length)];
			speakerBAudioSource.Play();
		}
		if (!speakerCAudioSource.isPlaying) {
			speakerCAudioSource.clip = (AudioClip)speakerCClips[Random.Range(0, speakerCClips.Length)];
			speakerCAudioSource.Play();
		}
    }
}
