from codeinterpreterapi import CodeInterpreterSession, File
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

async def main():
    # context manager for auto start/stop of the session
    async with CodeInterpreterSession() as session:
        while True:
            # define the user request
            user_request = input("Input your request: ") 
            # files = [
            #     File.from_path("examples/assets/iris.csv"),
            # ]

            # generate the response
            response = await session.generate_response(
                user_request, 
                # files=files,
            )
            print("AI: ", response.content)
            for file in response.files:
                file.show_image()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())