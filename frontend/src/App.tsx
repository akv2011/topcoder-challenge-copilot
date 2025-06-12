import React from 'react';
import { useState, useEffect, useCallback } from 'react';
import { Toaster, toast } from 'sonner';

// Type definitions
interface Message {
  role: 'user' | 'agent' | 'system';
  content: string;
}

interface Challenge {
  id: string;
  goal: string;
  status: string;
  platform_id?: string;
  dialogue_history: Message[];
  specification?: string;
  reasoning_traces?: string[];
  current_step_key?: string;
  created_at: string;
}

interface Platform {
  id: string;
  name: string;
  description?: string;
  schema_definition: string;
  created_at: string;
}

// API functions
const API_BASE = 'http://localhost:8000';

interface UploadResponse {
  message: string;
  challenge_id: string;
  filename: string;
  analysis_summary: string;
}

const api = {
  async createChallenge(goal: string): Promise<Challenge> {
    const response = await fetch(`${API_BASE}/challenges`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal }),
    });
    if (!response.ok) throw new Error('Failed to create challenge');
    return response.json();
  },

  async listChallenges(): Promise<Challenge[]> {
    const response = await fetch(`${API_BASE}/challenges`);
    if (!response.ok) throw new Error('Failed to fetch challenges');
    return response.json();
  },

  async getChallenge(id: string): Promise<Challenge> {
    const response = await fetch(`${API_BASE}/challenges/${id}`);
    if (!response.ok) throw new Error('Failed to fetch challenge');
    return response.json();
  },

  async sendMessage(challengeId: string, content: string): Promise<void> {
    const response = await fetch(`${API_BASE}/challenges/${challengeId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ challenge_id: challengeId, content }),
    });
    if (!response.ok) throw new Error('Failed to send message');
  },

  async listPlatforms(): Promise<Platform[]> {
    const response = await fetch(`${API_BASE}/platforms`);
    if (!response.ok) throw new Error('Failed to fetch platforms');
    return response.json();
  },

  async selectPlatform(challengeId: string, platformId: string): Promise<void> {
    const response = await fetch(`${API_BASE}/challenges/${challengeId}/platform`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ challenge_id: challengeId, platform_id: platformId }),
    });
    if (!response.ok) throw new Error('Failed to select platform');
  },

  async uploadFile(challengeId: string, file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('challenge_id', challengeId); // Ensure backend expects 'challenge_id'

    const response = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
      // Note: 'Content-Type' header is not set manually for FormData,
      // the browser sets it correctly with the boundary.
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Upload failed with no details' }));
      throw new Error(errorData.detail || 'File upload failed');
    }
    return response.json();
  },
};

// Components
function NewChallengeForm({ onChallengeCreated }: { onChallengeCreated: (id: string) => void }) {
  const [goal, setGoal] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim()) return;

    setIsLoading(true);
    try {
      const challenge = await api.createChallenge(goal);
      setGoal('');
      onChallengeCreated(challenge.id);
      toast.success('Challenge created! AI Copilot is ready to help.');
    } catch (error) {
      console.error('Failed to create challenge:', error);
      toast.error('Failed to create challenge');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 border rounded-lg shadow-sm bg-white">
      <h2 className="text-lg font-semibold mb-2 text-blue-600">Start a New Challenge</h2>
      <div className="flex flex-col gap-2">
        <textarea
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder="Describe your innovation challenge goal..."
          className="border rounded p-2 min-h-[80px] resize-none"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !goal.trim()}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-300"
        >
          {isLoading ? 'Creating...' : 'Start Challenge'}
        </button>
      </div>
    </form>
  );
}

function ChallengeList({ challenges, onSelectChallenge }: { 
  challenges: Challenge[]; 
  onSelectChallenge: (id: string) => void;
}) {
  if (challenges.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No challenges yet. Create your first challenge above!
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">Your Challenges</h2>
      {challenges.map((challenge) => (
        <div
          key={challenge.id}
          className="p-4 border rounded-lg shadow-sm bg-white cursor-pointer hover:bg-gray-50"
          onClick={() => onSelectChallenge(challenge.id)}
        >
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-lg font-semibold text-blue-700">{challenge.goal.substring(0, 100)}{challenge.goal.length > 100 ? '...' : ''}</h3>
              <p className="text-sm text-gray-500">Status: <span className="font-medium text-gray-700">{challenge.status}</span></p>
            </div>
            <span className="text-xs text-gray-400">
              {new Date(challenge.created_at).toLocaleDateString()}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ChatInterface({ challenge }: { challenge: Challenge }) {
  const [newMessage, setNewMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>(challenge.dialogue_history);

  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);


  // Poll for new messages
  useEffect(() => {
    setMessages(challenge.dialogue_history); // Update messages if challenge prop changes (e.g. new challenge selected)
  }, [challenge.dialogue_history]);

  useEffect(() => {
    const interval = setInterval(async () => {
      if (document.hidden) return; // Don't poll if tab is not active
      try {
        const updatedChallenge = await api.getChallenge(challenge.id);
        if (updatedChallenge.dialogue_history.length !== messages.length) {
          setMessages(updatedChallenge.dialogue_history);
        }
      } catch (error) {
        console.error('Failed to fetch updates:', error);
        // Potentially stop polling or show error to user if critical
      }
    }, 3000); // Increased polling interval slightly

    return () => clearInterval(interval);
  }, [challenge.id, messages.length]); // messages.length ensures re-evaluation if messages are updated by other means

  const handleFileUpload = async (file: File) => {
    if (!file) return;
    setIsUploading(true);
    
    // Add immediate "uploading" message to chat
    const uploadingMessage: Message = { 
      role: 'system', 
      content: `📤 Uploading file "${file.name}"... Processing with AI vision model.`
    };
    setMessages(prev => [...prev, uploadingMessage]);
    
    try {
      const result = await api.uploadFile(challenge.id, file);
      toast.success('File analyzed successfully!');
      
      // Update the uploading message with the result
      setMessages(prev => {
        const newMessages = [...prev];
        const lastIndex = newMessages.length - 1;
        
        // Replace the "uploading" message with the analysis result
        if (newMessages[lastIndex]?.role === 'system' && newMessages[lastIndex]?.content?.includes('📤 Uploading')) {
          newMessages[lastIndex] = {
            role: 'system',
            content: `📎 File "${result.filename}" uploaded and analyzed!\n\n**AI Analysis:**\n${result.analysis_summary}\n\n*The AI will now use this visual context to help define your challenge.*`
          };
        }
        
        return newMessages;
      });
      
      if(fileInputRef.current) {
        fileInputRef.current.value = ""; // Clear the file input
      }

    } catch (error: any) {
      console.error('File upload error:', error);
      toast.error(error.message || 'Failed to upload file');
      
      // Update the uploading message with error
      setMessages(prev => {
        const newMessages = [...prev];
        const lastIndex = newMessages.length - 1;
        
        if (newMessages[lastIndex]?.role === 'system' && newMessages[lastIndex]?.content?.includes('📤 Uploading')) {
          newMessages[lastIndex] = {
            role: 'system',
            content: `❌ Failed to upload file "${file.name}": ${error.message || 'Unknown error'}`
          };
        }
        
        return newMessages;
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file); // Immediately attempt upload
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMessage.trim()) return;

    const userMessage: Message = { role: 'user', content: newMessage };
    setMessages(prev => [...prev, userMessage]);
    const currentMessage = newMessage;
    setNewMessage('');
    setIsLoading(true);

    try {
      await api.sendMessage(challenge.id, currentMessage);
      // Message sent, polling will pick up the agent's response.
    } catch (error) {
      console.error('Failed to send message:', error);
      toast.error('Failed to send message');
      // Optionally, revert the optimistic message update or mark it as failed
      setMessages(prev => prev.filter(msg => msg !== userMessage));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] border rounded-lg bg-white shadow-md">
      <div className="flex-1 p-4 overflow-y-auto space-y-4 scroll-smooth">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`p-3 rounded-lg max-w-[80%] break-words ${
              message.role === 'user'
                ? 'bg-blue-500 text-white ml-auto'
                : message.role === 'agent'
                ? 'bg-gray-100 text-gray-800 mr-auto'
                : 'bg-yellow-50 text-yellow-700 border border-yellow-200 text-sm w-full text-center italic'
            }`}
          >
            <div className="whitespace-pre-wrap">
              {message.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="bg-gray-100 p-3 rounded-lg max-w-[80%] mr-auto">
            <span className="italic text-gray-500">Agent is thinking...</span>
          </div>
        )}
      </div>
      
      <form onSubmit={handleSubmit} className="p-4 border-t bg-gray-50">
        <div className="flex flex-col space-y-3">
          {/* File Upload Section */}
          <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-center space-x-3">
              <span className="text-blue-600 font-medium">📎 Upload Mockup or Image</span>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                className="text-sm file:mr-2 file:py-1 file:px-3 file:rounded file:border-0 file:bg-blue-500 file:text-white file:cursor-pointer hover:file:bg-blue-600"
                accept="image/*"
                disabled={isUploading}
              />
            </div>
            {isUploading && (
              <div className="flex items-center space-x-2 text-blue-600">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span className="text-sm font-medium">Processing with AI...</span>
              </div>
            )}
          </div>
          
          {/* Text Message Section */}
          <div className="flex space-x-2">
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 border rounded px-3 py-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={isLoading || isUploading}
            />
            <button
              type="submit"
              disabled={isLoading || isUploading || !newMessage.trim()}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-300 flex items-center space-x-2"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Sending...</span>
                </>
              ) : (
                <span>Send</span>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

function PlatformSelector({ challenge, onPlatformSelected }: { challenge: Challenge, onPlatformSelected: () => void }) {
  const [platforms, setPlatforms] = useState<Platform[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const loadPlatforms = async () => {
      try {
        const data = await api.listPlatforms();
        setPlatforms(data);
      } catch (error) {
        console.error('Failed to load platforms:', error);
        toast.error('Failed to load platforms');
      }
    };
    loadPlatforms();
  }, []);

  const handleSelectPlatform = async (platformId: string) => {
    setIsLoading(true);
    try {
      await api.selectPlatform(challenge.id, platformId);
      toast.success('Platform selected!');
      onPlatformSelected(); // Callback to refresh challenge details
    } catch (error) {
      console.error('Failed to select platform:', error);
      toast.error('Failed to select platform');
    } finally {
      setIsLoading(false);
    }
  };

  if (challenge.platform_id) {
    const selectedPlatform = platforms.find(p => p.id === challenge.platform_id);
    return (
      <div className="p-4 border rounded-lg bg-green-50 text-green-700 shadow-sm">
        <h3 className="font-medium text-lg">Platform Selected</h3>
        <p>{selectedPlatform?.name || 'Unknown Platform'}</p>
      </div>
    );
  }

  return (
    <div className="p-4 border rounded-lg bg-white shadow-sm">
      <h3 className="font-medium mb-3 text-lg text-gray-700">Select a Platform</h3>
      {platforms.length === 0 && <p className="text-gray-500">Loading platforms...</p>}
      <div className="space-y-2">
        {platforms.map((platform) => (
          <button
            key={platform.id}
            onClick={() => handleSelectPlatform(platform.id)}
            disabled={isLoading}
            className="w-full text-left p-3 border rounded-md hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          >
            <h4 className="font-semibold text-blue-600">{platform.name}</h4>
            <p className="text-sm text-gray-600">{platform.description}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

// Conceptual Checklist View Component
function ChecklistView({ challenge }: { challenge: Challenge }) {
  // This is a conceptual component.
  // In a real implementation, you would:
  // 1. Define what constitutes a "checklist item" (e.g., specific fields in the spec, stages of scoping).
  // 2. The backend would need to provide the status of these items.
  //    This could be derived from the challenge.specification or a dedicated checklist state.
  // 3. Fetch this data and render it.

  const checklistItems = [
    { id: 'goal', label: 'Initial Goal Defined', completed: !!challenge.goal },
    { id: 'scope', label: 'Scope Confirmed', completed: challenge.status !== 'scoping_goal' && challenge.status !== 'selecting_platform' }, // Example logic
    { id: 'platform', label: 'Platform Selected', completed: !!challenge.platform_id },
    { id: 'spec_details', label: 'Specification Details Being Defined', completed: challenge.status === 'defining_details' },
    // Add more items based on your challenge definition flow
  ];

  return (
    <div className="p-4 border rounded-lg bg-white shadow-sm">
      <h3 className="font-medium mb-3 text-lg text-gray-700">Challenge Progress Checklist</h3>
      <ul className="space-y-2">
        {checklistItems.map(item => (
          <li key={item.id} className="flex items-center">
            <span className={`w-5 h-5 rounded-full mr-3 flex items-center justify-center text-white ${item.completed ? 'bg-green-500' : 'bg-gray-300'}`}>
              {item.completed ? '✓' : ''}
            </span>
            <span className={item.completed ? 'text-gray-700' : 'text-gray-500'}>{item.label}</span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-gray-400 mt-3 italic">This is a conceptual checklist. Backend integration needed for full functionality.</p>
    </div>
  );
}

// Conceptual Challenge Spec Viewer Component
function ChallengeSpecViewer({ challenge }: { challenge: Challenge }) {
  // This is a conceptual component.
  // In a real implementation, you would:
  // 1. Fetch the challenge.specification (which is expected to be a JSON string).
  // 2. Parse the JSON.
  // 3. Render it in a user-friendly format.
  //    This might involve iterating over keys and values, or using a more structured approach
  //    if the spec schema is known.

  let specObject: any = null;
  try {
    if (challenge.specification) {
      specObject = JSON.parse(challenge.specification);
    }
  } catch (e) {
    console.error("Failed to parse specification JSON:", e);
  }

  return (
    <div className="p-4 border rounded-lg bg-white shadow-sm">
      <h3 className="font-medium mb-3 text-lg text-gray-700">Challenge Specification Viewer</h3>
      {specObject ? (
        <div className="space-y-2">
          {Object.entries(specObject).map(([key, value]) => (
            <div key={key}>
              <strong className="capitalize text-gray-600">{key.replace(/_/g, ' ')}:</strong>
              <pre className="bg-gray-50 p-2 rounded text-sm text-gray-800 whitespace-pre-wrap break-all">
                {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
              </pre>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-gray-500">No specification available or specification is not valid JSON.</p>
      )}
      {challenge.reasoning_traces && challenge.reasoning_traces.length > 0 && (
        <div className="mt-4">
          <h4 className="font-medium text-md text-gray-600 mb-1">Reasoning Traces:</h4>
          <pre className="bg-gray-50 p-2 rounded text-xs text-gray-700 whitespace-pre-wrap break-all">
            {JSON.stringify(challenge.reasoning_traces, null, 2)}
          </pre>
        </div>
      )}
      <p className="text-xs text-gray-400 mt-3 italic">This is a conceptual viewer. Backend provides the specification JSON.</p>
    </div>
  );
}


function ChallengeDetail({ challengeId, onBack, platforms }: { challengeId: string; onBack: () => void; platforms: Platform[] }) {
  const [challenge, setChallenge] = useState<Challenge | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadChallenge = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.getChallenge(challengeId);
      setChallenge(data);
    } catch (err: any) {
      console.error('Failed to load challenge:', err);
      setError(err.message || 'Failed to load challenge details.');
      toast.error(err.message || 'Failed to load challenge details.');
    } finally {
      setIsLoading(false);
    }
  }, [challengeId]);


  useEffect(() => {
    loadChallenge();
  }, [loadChallenge]);

  // Polling for challenge updates (e.g., status changes, spec updates)
  useEffect(() => {
    const interval = setInterval(() => {
      if (document.hidden) return;
      loadChallenge();
    }, 5000); // Poll every 5 seconds for overall challenge object changes
    return () => clearInterval(interval);
  }, [loadChallenge]);


  if (isLoading) {
    return <div className="text-center py-8 text-gray-600">Loading challenge details...</div>;
  }

  if (error) {
    return <div className="text-center py-8 text-red-500">Error: {error} <button onClick={loadChallenge} className="text-blue-500 underline ml-2">Try again</button></div>;
  }

  if (!challenge) {
    return <div className="text-center py-8 text-red-500">Challenge not found.</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="text-blue-500 hover:text-blue-700 flex items-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
          Back to Challenges
        </button>
        <div className="text-sm text-gray-500">
          Status: <span className="font-semibold text-gray-700">{challenge.status.replace(/_/g, ' ')}</span>
        </div>
      </div>

      <div>
        <h1 className="text-2xl font-bold text-gray-800 mb-1">Challenge: {challenge.goal}</h1>
        <p className="text-sm text-gray-500">ID: {challenge.id}</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="lg:col-span-1"> {/* Chat interface takes full width on smaller screens, half on larger */}
          <ChatInterface challenge={challenge} />
        </div>

        <div className="space-y-6 lg:col-span-1"> {/* Side panel for platform, checklist, spec */}
          {!challenge.platform_id && challenge.status === 'selecting_platform' && (
            <PlatformSelector challenge={challenge} onPlatformSelected={loadChallenge} />
          )}
          {challenge.platform_id && (
             <div className="p-4 border rounded-lg bg-green-50 text-green-700 shadow-sm">
              <h3 className="font-medium text-lg">Platform Selected</h3>
              <p>{platforms.find((p: Platform) => p.id === challenge.platform_id)?.name || 'Unknown Platform'}</p>
            </div>
          )}
          <ChecklistView challenge={challenge} />
          <ChallengeSpecViewer challenge={challenge} />
        </div>
      </div>
    </div>
  );
}

// Main App Component
function App() {
  const [selectedChallengeId, setSelectedChallengeId] = useState<string | null>(null);
  const [showNewChallengeForm, setShowNewChallengeForm] = useState(false);
  const [challenges, setChallenges] = useState<Challenge[]>([]);
  const [isLoadingChallenges, setIsLoadingChallenges] = useState(true);
  const [platforms, setPlatforms] = useState<Platform[]>([]); // Store platforms globally for access in ChallengeDetail

  useEffect(() => {
    const loadInitialData = async () => {
      setIsLoadingChallenges(true);
      try {
        const challengesData = await api.listChallenges();
        setChallenges(challengesData);
        const platformData = await api.listPlatforms(); // Load platforms once
        setPlatforms(platformData);
      } catch (error) {
        console.error('Failed to load initial data:', error);
        toast.error('Failed to load initial data');
      } finally {
        setIsLoadingChallenges(false);
      }
    };
    loadInitialData();
  }, []);

  const refreshChallenges = useCallback(async () => {
    try {
      const data = await api.listChallenges();
      setChallenges(data);
    } catch (error) {
      console.error('Failed to refresh challenges:', error);
      toast.error('Failed to refresh challenges');
    }
  }, []);

  const handleChallengeCreated = (challengeId: string) => {
    setSelectedChallengeId(challengeId);
    setShowNewChallengeForm(false);
    refreshChallenges(); // Refresh challenges list
  };

  const handleBackToList = () => {
    setSelectedChallengeId(null);
    refreshChallenges(); // Refresh challenges list
  };

  if (selectedChallengeId) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="container mx-auto px-4 py-8">
          <ChallengeDetail challengeId={selectedChallengeId} onBack={handleBackToList} platforms={platforms} />
        </div>
        <Toaster richColors position="top-right" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-blue-700">AI Challenge Copilot</h1>
          <p className="text-lg text-gray-600">Define and launch innovation challenges with AI assistance.</p>
        </div>

        <div className="max-w-2xl mx-auto space-y-6">
          {!showNewChallengeForm && (
            <button
              onClick={() => setShowNewChallengeForm(true)}
              className="w-full bg-green-500 text-white px-6 py-3 rounded-lg hover:bg-green-600 text-lg font-semibold shadow-md"
            >
              + Create New Challenge
            </button>
          )}
          {showNewChallengeForm && (
            <NewChallengeForm onChallengeCreated={handleChallengeCreated} />
          )}
          
          {isLoadingChallenges ? (
            <div className="text-center py-8 text-gray-500">Loading challenges...</div>
          ) : (
            <ChallengeList challenges={challenges} onSelectChallenge={setSelectedChallengeId} />
          )}
        </div>
      </div>
      <Toaster richColors position="top-right" />
    </div>
  );
}

export default App;
