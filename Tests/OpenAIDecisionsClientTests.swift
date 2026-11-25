import Foundation
import Testing
@testable import AXeCore

@Suite("OpenAI Decisions client")
struct OpenAIDecisionsClientTests {
    private let payload = TypeSafeSystemOneRequest(
        model: OpenAIDecisionsClient.defaultModel,
        state: .object(["screen": .string("e3|type|text-field|City||")]),
        questions: [
            "type_text_value_e3": TypeSafeChoiceQuestion(
                instructions: .object(["goal": .string("Enter «Café»")]),
                criteria: ["v1": .string("Café")]
            ),
            "operation": TypeSafeChoiceQuestion(criteria: [
                "TYPE_TEXT": .string("Enter text"),
                "DONE": .string("Finish")
            ])
        ]
    )

    @Test("Encodes OpenAI choices and returns the shared decision format")
    func translatesRequestAndResponse() async throws {
        let client = try OpenAIDecisionsClient(apiKey: "test-token") { request in
            #expect(request.url?.absoluteString == "https://api.openai.com/v1/decisions")
            #expect(request.httpMethod == "POST")
            #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-token")
            let body = try JSONDecoder().decode(TestRequest.self, from: #require(request.httpBody))
            #expect(body.model == "gpt-6-luna")
            #expect(body.input == #"{"screen":"e3|type|text-field|City||"}"#)
            #expect(body.questions.map(\.name) == ["operation", "type_text_value_e3"])
            #expect(body.questions.allSatisfy { $0.type == "choice" })
            #expect(body.questions[1].instructions == #"{"goal":"Enter «Café»"}"#)
            #expect(body.questions[1].choices == [
                .init(value: "v1", description: "Café"),
                .init(value: "__axe_no_match__", description: "The offered choice does not fit the goal or current state.")
            ])
            return try Self.response(
                request,
                """
                {"model":"gpt-6-luna","usage":{"input_tokens":25,"output_tokens":0},"answers":[
                  {"name":"operation","type":"choice","choice":"TYPE_TEXT","confidence":0.91,
                   "probabilities":[{"value":"TYPE_TEXT","probability":0.91},{"value":"DONE","probability":0.09}]},
                  {"name":"type_text_value_e3","type":"choice","choice":"v1","confidence":0.78,
                   "probabilities":[{"value":"v1","probability":0.78},{"value":"__axe_no_match__","probability":0.22}]}
                ]}
                """
            )
        }
        let result = try await client.decide(payload)
        #expect(result.answers["operation"]?.probabilities == ["TYPE_TEXT": 0.91, "DONE": 0.09])
        #expect(result.answers["type_text_value_e3"]?.probabilities == ["v1": 1])
        #expect(result.answers["type_text_value_e3"]?.confidence == 0.78)
        #expect(result.usage?.inputTokens == 25)
    }

    @Test("A refusal and a rejected sole literal leave the required answer missing")
    func omitsRefusalAndAbstention() async throws {
        let base = try #require(URL(string: "http://localhost:3119"))
        let client = try OpenAIDecisionsClient(apiKey: "test-token", baseURL: base) { request in
            #expect(request.url?.path == "/v1/decisions")
            return try Self.response(
                request,
                """
                {"model":"gpt-6-luna","answers":[
                  {"name":"operation","type":"refusal"},
                  {"name":"type_text_value_e3","type":"choice","choice":"__axe_no_match__","confidence":0.9,
                   "probabilities":[{"value":"v1","probability":0.1},{"value":"__axe_no_match__","probability":0.9}]}
                ]}
                """
            )
        }
        #expect(try await client.decide(payload).answers.isEmpty)
    }

    @Test("Answer names must match the OpenAI request order")
    func rejectsMismatchedAnswers() async throws {
        let client = try OpenAIDecisionsClient(apiKey: "test-token") { request in
            try Self.response(
                request,
                """
                {"model":"gpt-6-luna","answers":[
                  {"name":"type_text_value_e3","type":"refusal"},
                  {"name":"operation","type":"refusal"}
                ]}
                """
            )
        }
        await #expect(throws: OpenAIDecisionsError.self) {
            try await client.decide(payload)
        }
    }

    private static func response(_ request: URLRequest, _ body: String) throws -> (Data, HTTPURLResponse) {
        let url = try #require(request.url)
        let response = try #require(HTTPURLResponse(
            url: url, statusCode: 200, httpVersion: nil, headerFields: nil
        ))
        return (Data(body.utf8), response)
    }

    private struct TestRequest: Decodable {
        struct Question: Decodable {
            struct Choice: Decodable, Equatable {
                let value: String
                let description: String
            }

            let type: String
            let name: String
            let instructions: String
            let choices: [Choice]
        }

        let model: String
        let input: String
        let questions: [Question]
    }
}
