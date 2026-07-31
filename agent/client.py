import asyncio
import json
import re
import sys
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from langchain_ollama import ChatOllama

class NOCAgentEngine:
    def __init__(self):
        self.llm = ChatOllama(model="mistral", temperature=0)
        self.memory = []

    def get_ai_decision(self, user_input, available_tools):
        tools_info = []
        for t in available_tools:
            tools_info.append(f"- {t.name}: {t.description} \n  Schema: {t.inputSchema}")
        
        tools_text = "\n".join(tools_info)
        context_text = "\n".join(self.memory[-4:]) if self.memory else "No previous context"
        
        prompt = f"""You are an autonomous NOC AI Assistant.
Available Tools:
{tools_text}

Recent Context:
{context_text}

User Request: "{user_input}"

INSTRUCTIONS (Choose OPTION A or OPTION B):

OPTION A - If you need to use a tool to get data, you MUST output ONLY a JSON object in this exact format:
{{"tool_name": "name_of_tool", "arguments": {{"key": value}}}}

OPTION B - If you do not need a tool and want to talk to the user, output ONLY a JSON object in this exact format:
{{"response": "Your message here"}}

STRICT RULES:
- NEVER write code blocks (no javascript, no python).
- NEVER add explanations outside the JSON object.
- node_id and customer_id must be integers.
"""
        res = self.llm.invoke(prompt)
        return res.content

async def run_client():
    print("==================================================================")
    print("🤖 NEXLINK NOC AGENT (Powered by Dynamic LLM & ReAct Parsing)")
    print("==================================================================")

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server/server.py", "--transport", "stdio"]
    )

    engine = NOCAgentEngine()

    print("Connecting to MCP Server over Stdio...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("[HANDSHAKE SUCCESS] MCP Session Initialized.\n")
            
            tools_res = await session.list_tools()
            
            print("NOC Agent Ready. Type your query naturally.")
            print("To authenticate as admin, type: /auth token-admin-9988\n")
            
            while True:
                try:
                    user_input = input("Engineer: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input or user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break

                if user_input.startswith("/auth "):
                    token = user_input.split("/auth ")[1].strip()
                    print("Processing Authentication...")
                    t_res = await session.call_tool("authenticate_user", arguments={"api_token": token})
                    print(f"✅ [Server]: {t_res.content[0].text}\n")
                    engine.memory.append(f"System: User successfully authenticated. Server message: {t_res.content[0].text}")
                    tools_res = await session.list_tools() 
                    continue
                    
                elif user_input == "/resources":
                    print("\n[MCP RESOURCES] Reading policy files...")
                    r_list = await session.list_resources()
                    for r in r_list.resources:
                        c = await session.read_resource(r.uri)
                        print(f"📄 {r.name}:\n{c.contents[0].text[:150]}...\n")
                    continue

                print("🧠 AI is thinking...")
                engine.memory.append(f"User: {user_input}")
                
                ai_response = engine.get_ai_decision(user_input, tools_res.tools)
                
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                
                if json_match:
                    try:
                        decision = json.loads(json_match.group())
                        
                        if "tool_name" in decision:
                            t_name = decision["tool_name"]
                            t_args = decision.get("arguments", {})
                            
                            print(f"⚙️ [AI Action]: Using tool '{t_name}' with args {t_args}")
                            
                            # ==========================================
                            SENSITIVE_TOOLS = ["upgrade_bandwidth"] 
                            
                            approved = True
                            if t_name in SENSITIVE_TOOLS:
                                print(f"\n⚠️ [SECURITY ALERT]: AI requested a sensitive operation ({t_name}).")
                                print(f"   Requested Parameters: {t_args}")
                                
                                approval = input("   👉 Do you approve this action? (yes/no): ").strip().lower()
                                
                                if approval not in ['yes', 'y']:
                                    approved = False
                            
                            
                            if approved:
                                # ==========================================
                                # (Progress Tracking)
                                # ==========================================
                                if t_name == "run_network_diagnostic":
                                    print("   ⏳ Initiating deep network diagnostic...")
                                    for i in range(1, 5):
                                        print(f"   📊 [MCP PROGRESS] Phase {i}/4 ({i*25}%)")
                                        await asyncio.sleep(0.2)
                            # ==========================================

                                try:
                                    t_res = await session.call_tool(t_name, arguments=t_args)
                                    out_text = t_res.content[0].text if t_res.content else ""
                                except Exception as e:
                                    out_text = f"Error: {str(e)}س"
                            else:
                                out_text = "Operation cancelled: Human engineer denied approval."
                                print("❌ [Action Denied by Admin]")
                            # ==========================================
                                
                            print(f"✅ [Server Response]: {out_text}")
                            engine.memory.append(f"System Response: {out_text}")
                            
                            print("🧠 AI is summarizing...")
                            summary_prompt = f"The tool returned: {out_text}. Summarize this result for the user in one clear sentence."
                            summary_res = engine.llm.invoke(summary_prompt)
                            print(f"\n🤖 AI Assistant:\n {summary_res.content}\n")
                            engine.memory.append(f"AI: {summary_res.content}")
                            
                        elif "response" in decision:
                            print(f"\n🤖 AI Assistant:\n {decision['response']}\n")
                            engine.memory.append(f"AI: {decision['response']}")
                            
                    except json.JSONDecodeError:
                        print(f"\n🤖 AI Assistant:\n {ai_response}\n")
                else:
                    print(f"\n🤖 AI Assistant:\n {ai_response}\n")

if __name__ == "__main__":
    asyncio.run(run_client())