import AXeCore
import Testing
@testable import AXe

@Suite("Agent decision provider selection")
struct AgentDecisionConfigurationTests {
    @Test("Jev remains the default with its existing environment variables")
    func jevDefault() throws {
        let command = try AgentCommand.parse(["Check Settings", "--udid", "sim"])
        let configuration = try command.decisionConfiguration(environment: [
            "TYPESAFE_API_KEY": "test-token",
            "TYPESAFE_MODEL": "jev-test",
            "TYPESAFE_INPUT_USD_PER_MTOK": "0.125",
            "OPENAI_API_KEY": "unused-token"
        ])
        #expect(configuration.client is TypeSafeClient)
        #expect(configuration.model == "jev-test")
        #expect(configuration.inputRate == 0.125)
    }

    @Test("OpenAI Decisions uses its own key with the same arguments")
    func decisionsOptIn() throws {
        let command = try AgentCommand.parse(["Check Settings", "--udid", "sim"])
        let environment = ["AXE_AGENT_PROVIDER": "decisions", "OPENAI_API_KEY": "test-token"]
        let configuration = try command.decisionConfiguration(environment: environment)
        #expect(configuration.client is OpenAIDecisionsClient)
        #expect(configuration.model == OpenAIDecisionsClient.defaultModel)
        #expect(throws: (any Error).self) {
            try command.decisionConfiguration(environment: [
                "AXE_AGENT_PROVIDER": "decisions", "TYPESAFE_API_KEY": "unused-token"
            ])
        }
    }
}
