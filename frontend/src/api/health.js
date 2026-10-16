import client from "./client";

export const getHealth = async () => (await client.get("/health")).data;
