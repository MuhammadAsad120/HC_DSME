import streamlit as st
import os
import json
import datetime
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import openai

@dataclass
class PatientProfile:
    patient_id: str
    name: str
    dob: str
    primary_conditions: List[str]
    medications: List[str]
    provider: str
    care_plan: Dict[str, Any] = field(default_factory=dict)
    recent_labs: Dict[str, Any] = field(default_factory=dict)
    monitoring_devices: List[str] = field(default_factory=list)
    
    # RPM/CCM specific data
    last_readings: Dict[str, Any] = field(default_factory=dict)
    reading_trends: Dict[str, List[Any]] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@dataclass
class ConversationEntry:
    timestamp: str
    role: str  # "user" or "assistant"
    content: str
    flags: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class PatientMemory:
    def __init__(self, storage_dir: str = "patient_data"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def _get_patient_file(self, patient_id: str) -> str:
        return os.path.join(self.storage_dir, f"{patient_id}.json")
    
    def load_patient(self, patient_id: str) -> Dict[str, Any]:
        file_path = self._get_patient_file(patient_id)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return {
            "profile": {},
            "conversation_history": []
        }
    
    def save_patient(self, patient_id: str, data: Dict[str, Any]) -> None:
        with open(self._get_patient_file(patient_id), 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_profile(self, patient_id: str) -> Optional[PatientProfile]:
        data = self.load_patient(patient_id)
        if data.get("profile"):
            return PatientProfile.from_dict(data["profile"])
        return None
    
    def save_profile(self, profile: PatientProfile) -> None:
        data = self.load_patient(profile.patient_id)
        data["profile"] = profile.to_dict()
        self.save_patient(profile.patient_id, data)
    
    def add_conversation_entry(self, patient_id: str, entry: ConversationEntry) -> None:
        data = self.load_patient(patient_id)
        if "conversation_history" not in data:
            data["conversation_history"] = []
        data["conversation_history"].append(entry.to_dict())
        self.save_patient(patient_id, data)
    
    def get_conversation_history(self, patient_id: str, count: int = 10) -> List[ConversationEntry]:
        data = self.load_patient(patient_id)
        history = data.get("conversation_history", [])
        entries = [ConversationEntry.from_dict(e) for e in history[-count:]]
        return entries
    
    def update_readings(self, patient_id: str, reading_type: str, value: Any) -> None:
        profile = self.get_profile(patient_id)
        if not profile:
            return
        
        # Update last reading
        profile.last_readings[reading_type] = {
            "value": value,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add to trend data
        if reading_type not in profile.reading_trends:
            profile.reading_trends[reading_type] = []
        
        profile.reading_trends[reading_type].append({
            "value": value,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Keep only last 30 readings
        profile.reading_trends[reading_type] = profile.reading_trends[reading_type][-30:]
        
        self.save_profile(profile)


class DynamicHealthcareConversation:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.memory = PatientMemory()
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.staff_name = "Fanny"
        self.staff_role = "MA"
        self.current_patient_id = None
        self.conversation_stage = "start"
        
        # System prompt for the healthcare conversation
        self.system_prompt = """
You are {staff_name}, a {staff_role} at Dr. {provider}'s office, conducting a monthly CCM/RPM (Chronic Care Management/Remote Patient Monitoring) check-in with the patient.

You should follow a conversational flow that includes:
1. Simple greeting
2. General check-in
3. Condition-specific questions
4. Medication review
5. Adherence verification
6. Lifestyle factors
7. Device monitoring check
8. Scheduling and closing

IMPORTANT GUIDELINES:
- Be warm and empathetic but professional
- Ask leading questions that encourage detailed responses
- Reference previous information to show continuity of care
- Ask ONE question at a time - don't overwhelm the patient
- Your responses should be concise (2-4 sentences)
- If you notice concerning symptoms or readings, make appropriate follow-up questions
- Format your responses as "{staff_name} ({staff_role}): [your message]"

CURRENT CONVERSATION STAGE: {conversation_stage}

PATIENT PROFILE:
Name: {name}
DOB: {dob}
Primary Conditions: {conditions}
Current Medications: {medications}
Care Plan: {care_plan}
Recent Labs: {recent_labs}
Monitoring Devices: {monitoring_devices}
Recent Readings: {recent_readings}

CONVERSATION HISTORY:
{conversation_history}
"""
    
    def create_or_update_patient(self, patient_data: Dict[str, Any]) -> str:
        """Create or update a patient profile"""
        patient_id = patient_data.get("patient_id", str(hash(patient_data.get("name", "") + patient_data.get("dob", ""))))
        
        existing_profile = self.memory.get_profile(patient_id)
        if existing_profile:
            # Update existing profile
            for key, value in patient_data.items():
                if hasattr(existing_profile, key):
                    setattr(existing_profile, key, value)
            profile = existing_profile
        else:
            # Create new profile
            profile = PatientProfile(
                patient_id=patient_id,
                name=patient_data.get("name", ""),
                dob=patient_data.get("dob", ""),
                primary_conditions=patient_data.get("primary_conditions", []),
                medications=patient_data.get("medications", []),
                provider=patient_data.get("provider", ""),
                care_plan=patient_data.get("care_plan", {}),
                recent_labs=patient_data.get("recent_labs", {}),
                monitoring_devices=patient_data.get("monitoring_devices", [])
            )
        
        self.memory.save_profile(profile)
        return patient_id
    
    def _format_conversation_history(self, history: List[ConversationEntry]) -> str:
        """Format the conversation history for the prompt"""
        if not history:
            return "No previous conversation."
        
        formatted = []
        for entry in history:
            formatted.append(f"{entry.role}: {entry.content}")
        
        return "\n".join(formatted)
    
    def _format_care_plan(self, care_plan: Dict[str, Any]) -> str:
        """Format the care plan for the prompt"""
        if not care_plan:
            return "No specific care plan recorded."
        
        formatted = []
        for key, value in care_plan.items():
            formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted)
    
    def _format_labs(self, labs: Dict[str, Any]) -> str:
        """Format the recent labs for the prompt"""
        if not labs:
            return "No recent labs recorded."
        
        formatted = []
        for key, value in labs.items():
            formatted.append(f"- {key}: {value}")
        
        return "\n".join(formatted)
    
    def _format_readings(self, last_readings: Dict[str, Any]) -> str:
        """Format the recent readings for the prompt"""
        if not last_readings:
            return "No recent readings recorded."
        
        formatted = []
        for key, data in last_readings.items():
            formatted.append(f"- {key}: {data['value']} (recorded on {data['timestamp'][:10]})")
        
        return "\n".join(formatted)
    
    def _determine_conversation_stage(self, history: List[ConversationEntry]) -> str:
        """Determine the current stage of the conversation based on history"""
        assistant_messages = [e for e in history if e.role == "assistant"]
        
        if not assistant_messages:
            return "greeting"
        elif len(assistant_messages) <= 2:
            return "general_check_in"
        elif len(assistant_messages) <= 4:
            return "condition_specific"
        elif len(assistant_messages) <= 6:
            return "medication_review"
        elif len(assistant_messages) <= 8:
            return "adherence"
        elif len(assistant_messages) <= 10:
            return "lifestyle"
        elif len(assistant_messages) <= 12:
            return "device_monitoring"
        else:
            return "closing"
    
    def _check_for_alerts(self, response: str, profile: PatientProfile) -> List[str]:
        """Check for any medical alerts based on the patient response"""
        alerts = []
        
        # Define alert keywords
        severe_symptom_keywords = [
            "severe pain", "unbearable", "excruciating", 
            "chest pain", "can't breathe", "difficulty breathing",
            "passing out", "fainted", "unconscious",
            "emergency", "ER", "hospital"
        ]
        
        # Check for severe symptoms
        response_lower = response.lower()
        for keyword in severe_symptom_keywords:
            if keyword in response_lower:
                alerts.append(f"Potential urgent symptom mentioned: {keyword}")
        
        # Check BP readings
        if "blood_pressure" in profile.last_readings:
            bp = profile.last_readings["blood_pressure"]["value"]
            if isinstance(bp, str) and "/" in bp:
                try:
                    systolic, diastolic = map(int, bp.split("/"))
                    if systolic > 160:
                        alerts.append(f"High systolic BP: {systolic}")
                    if diastolic > 100:
                        alerts.append(f"High diastolic BP: {diastolic}")
                except (ValueError, IndexError):
                    pass
        
        # Check weight change
        if "weight" in profile.reading_trends and len(profile.reading_trends["weight"]) > 7:
            recent = profile.reading_trends["weight"][-1]["value"]
            week_ago = profile.reading_trends["weight"][-7]["value"]
            change = recent - week_ago
            if abs(change) >= 2:
                direction = "gain" if change > 0 else "loss"
                alerts.append(f"Weight {direction} of {abs(change)} lbs in 7 days")
        
        return alerts
    
    def start_conversation(self, patient_id: str) -> str:
        """Start a new conversation with a patient"""
        self.current_patient_id = patient_id
        self.conversation_stage = "greeting"
        
        profile = self.memory.get_profile(patient_id)
        if not profile:
            return "Patient profile not found. Please create a profile first."
        
        # Generate the initial greeting using OpenAI
        try:
            system_content = self.system_prompt.format(
                staff_name=self.staff_name,
                staff_role=self.staff_role,
                provider=profile.provider,
                conversation_stage=self.conversation_stage,
                name=profile.name,
                dob=profile.dob,
                conditions=", ".join(profile.primary_conditions),
                medications=", ".join(profile.medications),
                care_plan=self._format_care_plan(profile.care_plan),
                recent_labs=self._format_labs(profile.recent_labs),
                monitoring_devices=", ".join(profile.monitoring_devices),
                recent_readings=self._format_readings(profile.last_readings),
                conversation_history="This is the start of the conversation."
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": "START_CONVERSATION"}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            assistant_message = response.choices[0].message.content
            
            # Save the assistant's message
            entry = ConversationEntry(
                timestamp=datetime.datetime.now().isoformat(),
                role="assistant",
                content=assistant_message
            )
            self.memory.add_conversation_entry(patient_id, entry)
            
            return assistant_message
            
        except Exception as e:
            return f"Error starting conversation: {str(e)}"
    
    def process_response(self, patient_response: str) -> str:
        """Process the patient's response and generate the next message"""
        if not self.current_patient_id:
            return "No active conversation. Please start a new conversation first."
        
        profile = self.memory.get_profile(self.current_patient_id)
        if not profile:
            return "Patient profile not found."
        
        # Save the user's message
        user_entry = ConversationEntry(
            timestamp=datetime.datetime.now().isoformat(),
            role="user",
            content=patient_response
        )
        self.memory.add_conversation_entry(self.current_patient_id, user_entry)
        
        # Check for any medical alerts
        alerts = self._check_for_alerts(patient_response, profile)
        if alerts:
            user_entry.flags = alerts
            # Re-save the entry with flags
            self.memory.add_conversation_entry(self.current_patient_id, user_entry)
        
        # Get conversation history
        history = self.memory.get_conversation_history(self.current_patient_id)
        
        # Determine the conversation stage
        self.conversation_stage = self._determine_conversation_stage(history)
        
        # Format history for the prompt
        formatted_history = self._format_conversation_history(history)
        
        # Generate the next message using OpenAI
        try:
            system_content = self.system_prompt.format(
                staff_name=self.staff_name,
                staff_role=self.staff_role,
                provider=profile.provider,
                conversation_stage=self.conversation_stage,
                name=profile.name,
                dob=profile.dob,
                conditions=", ".join(profile.primary_conditions),
                medications=", ".join(profile.medications),
                care_plan=self._format_care_plan(profile.care_plan),
                recent_labs=self._format_labs(profile.recent_labs),
                monitoring_devices=", ".join(profile.monitoring_devices),
                recent_readings=self._format_readings(profile.last_readings),
                conversation_history=formatted_history
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            assistant_message = response.choices[0].message.content
            
            # Save the assistant's message
            entry = ConversationEntry(
                timestamp=datetime.datetime.now().isoformat(),
                role="assistant",
                content=assistant_message
            )
            self.memory.add_conversation_entry(self.current_patient_id, entry)
            
            return assistant_message
            
        except Exception as e:
            return f"Error processing response: {str(e)}"
    
    def update_patient_reading(self, reading_type: str, value: Any) -> None:
        """Update a patient's health reading"""
        if not self.current_patient_id:
            return
        
        self.memory.update_readings(self.current_patient_id, reading_type, value)


# Streamlit App
def main():
    st.set_page_config(page_title="Healthcare Chatbot", page_icon="🏥", layout="wide")
    
    st.title("Healthcare CCM/RPM Check-in Chatbot")
    st.markdown("---")
    
    # Initialize session state
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'healthcare_chat' not in st.session_state:
        st.session_state.healthcare_chat = None
    if 'patient_id' not in st.session_state:
        st.session_state.patient_id = None
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            if api_key:
                st.session_state.healthcare_chat = DynamicHealthcareConversation(api_key=api_key)
        
        # Model selection
        model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)
        if st.session_state.healthcare_chat and model:
            st.session_state.healthcare_chat.model = model
        
        st.markdown("---")
        
        # Patient Profile Creation
        st.header("Patient Profile")
        
        with st.form("patient_form"):
            name = st.text_input("Patient Name")
            dob = st.text_input("Date of Birth")
            conditions = st.text_input("Primary Conditions (comma separated)")
            medications = st.text_input("Medications (comma separated)")
            provider = st.text_input("Provider Name")
            
            # Optional Fields
            with st.expander("Care Plan (Optional)"):
                diet_type = st.text_input("Diet Type")
                exercise_plan = st.text_input("Exercise Plan")
                lifestyle_goal = st.text_input("Lifestyle Goal")
            
            with st.expander("Recent Labs (Optional)"):
                lab1_name = st.text_input("Lab Test 1 Name")
                lab1_value = st.text_input("Lab Test 1 Value")
                lab2_name = st.text_input("Lab Test 2 Name")
                lab2_value = st.text_input("Lab Test 2 Value")
            
            with st.expander("Monitoring Devices (Optional)"):
                devices = st.text_input("Devices (comma separated)")
            
            with st.expander("Readings (Optional)"):
                bp = st.text_input("Blood Pressure (systolic/diastolic)")
                weight = st.number_input("Weight (lbs)", min_value=0.0, value=0.0)
            
            submit_button = st.form_submit_button("Create/Update Patient")
            
            if submit_button and name and dob and st.session_state.healthcare_chat:
                # Process form data
                patient_data = {
                    "name": name,
                    "dob": dob,
                    "primary_conditions": [c.strip() for c in conditions.split(",")] if conditions else [],
                    "medications": [m.strip() for m in medications.split(",")] if medications else [],
                    "provider": provider,
                }
                
                # Add optional fields if provided
                care_plan = {}
                if diet_type:
                    care_plan["diet_type"] = diet_type
                if exercise_plan:
                    care_plan["exercise_plan"] = exercise_plan
                if lifestyle_goal:
                    care_plan["lifestyle_goal"] = lifestyle_goal
                
                if care_plan:
                    patient_data["care_plan"] = care_plan
                
                recent_labs = {}
                if lab1_name and lab1_value:
                    recent_labs[lab1_name] = lab1_value
                if lab2_name and lab2_value:
                    recent_labs[lab2_name] = lab2_value
                
                if recent_labs:
                    patient_data["recent_labs"] = recent_labs
                
                if devices:
                    patient_data["monitoring_devices"] = [d.strip() for d in devices.split(",")]
                
                # Create or update patient
                patient_id = st.session_state.healthcare_chat.create_or_update_patient(patient_data)
                st.session_state.patient_id = patient_id
                
                # Add readings if provided
                if bp:
                    st.session_state.healthcare_chat.update_patient_reading("blood_pressure", bp)
                
                if weight > 0:
                    st.session_state.healthcare_chat.update_patient_reading("weight", weight)
                
                st.success(f"Patient profile created/updated successfully!")
        
        # Start conversation button
        if st.session_state.patient_id and not st.session_state.conversation_started:
            if st.button("Start Conversation"):
                with st.spinner("Starting conversation..."):
                    first_message = st.session_state.healthcare_chat.start_conversation(st.session_state.patient_id)
                    st.session_state.messages.append({"role": "assistant", "content": first_message})
                    st.session_state.conversation_started = True
                    st.rerun()  # Updated from experimental_rerun
    
    # Main chat interface
    if st.session_state.conversation_started:
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.text_area(
    "You",
    value=message["content"],
    disabled=True,
    key=f"user_{i}"
)
                else:
                    st.text_area("Assistant", value=message["content"], height=100, disabled=True, key=f"assistant_{i}")
        
        # User input
        user_input = st.text_area("Your response:", height=100, key="user_input")
        if st.button("Send"):
            if user_input:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Get assistant response
                with st.spinner("Generating response..."):
                    assistant_response = st.session_state.healthcare_chat.process_response(user_input)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
                # Clear input and rerun
                st.rerun()  # Updated from experimental_rerun
    elif not st.session_state.api_key:
        st.info("Please enter your OpenAI API key in the sidebar to begin.")
    elif not st.session_state.patient_id:
        st.info("Please create a patient profile in the sidebar to begin.")
    else:
        st.info("Click 'Start Conversation' in the sidebar to begin the check-in.")


if __name__ == "__main__":
    main()