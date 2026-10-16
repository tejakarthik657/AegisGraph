import client from "./client";

export const getTransactions = async (params = {}) =>
  (await client.get("/transactions", { params })).data;
export const getTransaction = async (id) =>
  (await client.get(`/transactions/${id}`)).data;
